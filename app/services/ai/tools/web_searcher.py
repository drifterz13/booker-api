from langchain_core.language_models import BaseChatModel


class WebSearcher:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model.bind_tools([{"type": "web_search"}], tool_choice="required")

    def search(self, query: str) -> str:
        response = self._model.invoke(
            f"Search the web for: {query}\nInclude source URLs in your answer."
        )
        urls: list[str] = []
        for block in response.content_blocks:
            if block["type"] != "text":
                continue

            for citation in block.get("annotations", []):
                if citation["type"] != "citation":
                    continue

                url = citation.get("url")
                if url and url not in urls:
                    urls.append(url)

        if urls:
            sources = "\n".join(
                f"- [Source {index}]({url})" for index, url in enumerate(urls, 1)
            )
            return f"{response.text}\n\nSources:\n{sources}"
        return response.text
