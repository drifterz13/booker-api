from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from ...core.config import StorageConfig


@dataclass(frozen=True, slots=True)
class StoredObject:
    object_key: str
    size: int
    last_modified: datetime


class ObjectStorage:
    """Synchronous access to books and images in an existing S3-compatible bucket."""

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

    def start_multipart(self, *, object_key: str) -> str:
        response = self._client.create_multipart_upload(
            Bucket=self._bucket_name, Key=object_key, ContentType="application/pdf"
        )
        return response["UploadId"]

    def presign_part(self, *, object_key: str, upload_id: str, part_number: int) -> str:
        """Allow a client to PUT one part directly to storage for 15 minutes."""
        return self._client.generate_presigned_url(
            "upload_part",
            Params={
                "Bucket": self._bucket_name,
                "Key": object_key,
                "UploadId": upload_id,
                "PartNumber": part_number,
            },
            ExpiresIn=900,
            HttpMethod="PUT",
        )

    def complete_multipart(
        self, *, object_key: str, upload_id: str, parts: Sequence[tuple[int, str]]
    ) -> str:
        response = self._client.complete_multipart_upload(
            Bucket=self._bucket_name,
            Key=object_key,
            UploadId=upload_id,
            MultipartUpload={
                "Parts": [
                    {"PartNumber": number, "ETag": etag}
                    for number, etag in sorted(parts)
                ]
            },
        )
        return response["ETag"]

    def abort_multipart(self, *, object_key: str, upload_id: str) -> None:
        self._client.abort_multipart_upload(
            Bucket=self._bucket_name, Key=object_key, UploadId=upload_id
        )

    def upload(
        self,
        source: BinaryIO,
        *,
        object_key: str,
        content_type: str = "application/pdf",
    ) -> None:
        """Upload from the stream's current position to a caller-owned key.

        Use a unique key such as ``books/<book_id>.pdf`` for each book. The
        caller handles PDF validation, checksums, and database persistence.
        Storage errors propagate so a failed upload cannot be marked successful.
        """
        self._client.upload_fileobj(
            source,
            self._bucket_name,
            object_key,
            ExtraArgs={"ContentType": content_type},
        )

    def presign_download(self, *, object_key: str) -> str:
        """Allow a client to download an object for 15 minutes."""
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket_name, "Key": object_key},
            ExpiresIn=900,
            HttpMethod="GET",
        )

    def download(self, *, object_key: str, destination: Path) -> None:
        """Download a PDF to a caller-owned local file."""
        self._client.download_file(self._bucket_name, object_key, str(destination))

    def size(self, *, object_key: str) -> int:
        """Read the size of a completed object without downloading it."""
        try:
            return self._client.head_object(Bucket=self._bucket_name, Key=object_key)[
                "ContentLength"
            ]
        except ClientError as error:
            if error.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                # HEAD does not distinguish a missing object from a missing bucket.
                self._client.head_bucket(Bucket=self._bucket_name)
                raise FileNotFoundError("Completed upload does not exist") from error
            raise

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
