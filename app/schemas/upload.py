from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, model_validator

from ..services.books import validate_book_filename

ObjectKey = Annotated[
    str,
    Field(
        pattern=r"^books/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.pdf$"
    ),
]
PartNumber = Annotated[int, Field(ge=1, le=10000)]


class UploadCreate(BaseModel):
    filename: Annotated[
        str, Field(max_length=255), AfterValidator(validate_book_filename)
    ]


class UploadPublic(BaseModel):
    upload_id: str
    object_key: str


class PartUrlsRequest(BaseModel):
    object_key: ObjectKey
    part_numbers: list[PartNumber] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_parts(self) -> "PartUrlsRequest":
        if len(set(self.part_numbers)) != len(self.part_numbers):
            raise ValueError("Part numbers must be unique")
        return self


class PartUrl(BaseModel):
    part_number: int
    url: str
    expires_in: int = 900


class CompletedPart(BaseModel):
    part_number: PartNumber
    etag: str = Field(min_length=1, max_length=256)


class UploadComplete(BaseModel):
    object_key: ObjectKey
    parts: list[CompletedPart] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def unique_parts(self) -> "UploadComplete":
        numbers = [part.part_number for part in self.parts]
        if len(set(numbers)) != len(numbers):
            raise ValueError("Part numbers must be unique")
        return self


class UploadCompleted(BaseModel):
    object_key: str
    etag: str
