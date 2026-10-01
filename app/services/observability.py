"""App-owned tracing; exporting traces never blocks an AI request."""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager

from openinference.instrumentation import (
    TraceConfig,
    get_attributes_from_context,
    using_attributes,
)
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import NoOpTracerProvider, Span
from pydantic_ai.models.instrumented import InstrumentationSettings

from ..core.config import ObservabilityConfig
from ..models.search import SearchHit

logger = logging.getLogger(__name__)


class PhoenixSpanProcessor(BatchSpanProcessor):
    """Export SDK embeddings; Pydantic AI owns model-call usage."""

    def on_end(self, span: ReadableSpan) -> None:
        if (
            span.instrumentation_scope is not None
            and span.instrumentation_scope.name
            == "openinference.instrumentation.openai"
            and span.attributes is not None
            and span.attributes.get("openinference.span.kind") == "LLM"
        ):
            return

        attributes = span.attributes or {}
        if attributes.get("openinference.span.kind") == "EMBEDDING" and (
            model := attributes.get("embedding.model_name")
        ):
            # Phoenix only calculates costs for LLM spans. Preserve the operation
            # marker and embedding attributes while adapting the exported kind.
            span._attributes = {
                **attributes,
                "openinference.span.kind": "LLM",
                "llm.model_name": model,
                "booker.operation": "embedding",
            }
        super().on_end(span)


class _SpanProcessor(OpenInferenceSpanProcessor):
    def on_start(self, span, parent_context=None) -> None:
        span.set_attributes(dict(get_attributes_from_context()))


class Observability:
    def __init__(self, provider: TracerProvider | None = None) -> None:
        self.provider = provider
        self.tracer = (provider or NoOpTracerProvider()).get_tracer("booker")
        self.instrumentation = (
            InstrumentationSettings(
                tracer_provider=provider, include_binary_content=False
            )
            if provider is not None
            else None
        )
        self._instrumentor: OpenAIInstrumentor | None = None

    @contextmanager
    def chat(self, *, session_id: str, book_id: str, index_id: str) -> Iterator[Span]:
        with (
            using_attributes(
                session_id=session_id,
                metadata={"book_id": book_id, "index_id": index_id},
            ),
            self.tracer.start_as_current_span(
                "booker.chat", attributes={"openinference.span.kind": "CHAIN"}
            ) as span,
        ):
            yield span

    @contextmanager
    def retrieval(self, query: str, *, book_id: str, index_id: str) -> Iterator[Span]:
        with self.tracer.start_as_current_span(
            "booker.retrieve",
            attributes={
                "openinference.span.kind": "RETRIEVER",
                "input.value": query,
                "metadata": json.dumps({"book_id": book_id, "index_id": index_id}),
            },
        ) as span:
            yield span

    @staticmethod
    def record_hits(span: Span, hits: list[SearchHit]) -> None:
        if not span.is_recording():
            return
        span.set_attribute("booker.retrieval.result_count", len(hits))
        for i, hit in enumerate(hits):
            prefix = f"retrieval.documents.{i}.document"
            span.set_attribute(f"{prefix}.id", str(hit.chunk_id))
            span.set_attribute(f"{prefix}.content", hit.text)
            span.set_attribute(f"{prefix}.score", 1 - hit.distance)
            span.set_attribute(
                f"{prefix}.metadata",
                json.dumps(
                    {
                        "index_id": str(hit.index_id),
                        "pdf_pages": [page + 1 for page in hit.pages],
                        "section_path": list(hit.path),
                        "cosine_distance": hit.distance,
                    }
                ),
            )

    def close(self) -> None:
        if self._instrumentor is not None:
            self._instrumentor.uninstrument()
            self._instrumentor = None
        if self.provider is not None:
            self.provider.force_flush(timeout_millis=5000)
            self.provider.shutdown()
            self.provider = None


NO_OBSERVABILITY = Observability()


def create_observability(config: ObservabilityConfig) -> Observability:
    if not config.phoenix_collector_endpoint:
        return NO_OBSERVABILITY
    provider = None
    try:
        instrumentor = OpenAIInstrumentor()
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": "booker-api",
                    "openinference.project.name": config.phoenix_project_name,
                }
            ),
            shutdown_on_exit=False,
        )
        observability = Observability(provider)
        provider.add_span_processor(_SpanProcessor())
        provider.add_span_processor(
            PhoenixSpanProcessor(
                OTLPSpanExporter(
                    endpoint=f"{config.phoenix_collector_endpoint}/v1/traces",
                    timeout=2,
                ),
                schedule_delay_millis=1000,
            )
        )
        instrumentor.instrument(
            tracer_provider=provider,
            config=TraceConfig(hide_embedding_vectors=True, hide_outputs=True),
        )
        observability._instrumentor = instrumentor
        return observability
    except Exception:
        logger.exception("Tracing setup failed; continuing without observability")
        if provider is not None:
            provider.shutdown()
        return NO_OBSERVABILITY
