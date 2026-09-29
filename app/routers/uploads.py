from typing import Annotated
from uuid import uuid4

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException, Path, Query, Response, status

from ..dependencies import StorageDep
from ..schemas.upload import (
    ObjectKey,
    PartUrl,
    PartUrlsRequest,
    UploadComplete,
    UploadCompleted,
    UploadCreate,
    UploadPublic,
)

router = APIRouter(prefix="/uploads", tags=["uploads"])
UploadId = Annotated[str, Path(min_length=1, max_length=1024)]


def storage_error(error: BotoCoreError | ClientError) -> HTTPException:
    if isinstance(error, ClientError):
        code = error.response["Error"]["Code"]
        if code == "NoSuchUpload":
            return HTTPException(404, "Upload does not exist or is already finished")
        if code in {
            "InvalidPart",
            "InvalidPartOrder",
            "EntityTooSmall",
            "InvalidRequest",
            "NoSuchKey",  # RustFS reports a missing uploaded part with this code.
        }:
            return HTTPException(400, "Invalid upload parts")
    return HTTPException(502, "Could not access upload storage")


@router.post("", status_code=status.HTTP_201_CREATED)
def start_upload(payload: UploadCreate, storage: StorageDep) -> UploadPublic:
    """Start a PDF upload with a server-generated object key."""
    object_key = f"books/{uuid4()}.pdf"
    try:
        upload_id = storage.start_multipart(object_key=object_key)
    except (BotoCoreError, ClientError) as error:
        raise storage_error(error) from error
    return UploadPublic(upload_id=upload_id, object_key=object_key)


@router.post("/{upload_id}/parts")
def sign_parts(
    upload_id: UploadId, payload: PartUrlsRequest, storage: StorageDep
) -> list[PartUrl]:
    """Return PUT URLs; the client sends PDF bytes directly to storage."""
    try:
        return [
            PartUrl(
                part_number=number,
                url=storage.presign_part(
                    object_key=payload.object_key,
                    upload_id=upload_id,
                    part_number=number,
                ),
            )
            for number in payload.part_numbers
        ]
    except (BotoCoreError, ClientError) as error:
        raise storage_error(error) from error


@router.post("/{upload_id}/complete")
def complete_upload(
    upload_id: UploadId, payload: UploadComplete, storage: StorageDep
) -> UploadCompleted:
    """Assemble the uploaded parts using the ETags returned by storage."""
    try:
        etag = storage.complete_multipart(
            object_key=payload.object_key,
            upload_id=upload_id,
            parts=[(part.part_number, part.etag) for part in payload.parts],
        )
    except (BotoCoreError, ClientError) as error:
        raise storage_error(error) from error
    return UploadCompleted(object_key=payload.object_key, etag=etag)


@router.delete("/{upload_id}", status_code=status.HTTP_204_NO_CONTENT)
def abort_upload(
    upload_id: UploadId,
    object_key: Annotated[ObjectKey, Query()],
    storage: StorageDep,
) -> Response:
    try:
        storage.abort_multipart(object_key=object_key, upload_id=upload_id)
    except (BotoCoreError, ClientError) as error:
        raise storage_error(error) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)
