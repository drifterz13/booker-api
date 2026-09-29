from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import boto3
from botocore.client import Config as BotoConfig

from ...core.config import StorageConfig


@dataclass(frozen=True, slots=True)
class StoredObject:
    object_key: str
    size: int
    last_modified: datetime


class ObjectStorage:
    """Synchronous access to PDF objects in an existing S3-compatible bucket."""

    def __init__(self, *, config: StorageConfig) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=config.s3_endpoint_url,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_key,
            region_name=config.s3_region_name,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )
        self._bucket_name = config.s3_bucket_name

    def close(self) -> None:
        self._client.close()

    def upload(self, source: BinaryIO, *, object_key: str) -> None:
        """Upload from the stream's current position to a caller-owned key.

        Use a unique key such as ``books/<book_id>.pdf`` for each book. The
        caller handles PDF validation, checksums, and database persistence.
        Storage errors propagate so a failed upload cannot be marked successful.
        """
        self._client.upload_fileobj(
            source,
            self._bucket_name,
            object_key,
            ExtraArgs={"ContentType": "application/pdf"},
        )

    def download(self, *, object_key: str, destination: Path) -> None:
        """Download a PDF to a caller-owned local file."""
        self._client.download_file(self._bucket_name, object_key, str(destination))

    def list(self, *, prefix: str = "books/") -> list[StoredObject]:
        """List object metadata across all pages, without downloading PDFs."""
        paginator = self._client.get_paginator("list_objects_v2")
        return [
            StoredObject(
                object_key=item["Key"],
                size=item["Size"],
                last_modified=item["LastModified"],
            )
            for page in paginator.paginate(Bucket=self._bucket_name, Prefix=prefix)
            for item in page.get("Contents", [])
        ]
