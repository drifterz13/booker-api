import unittest

import httpx
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import get_storage
from app.routers.uploads import router
from app.services.storage.storage import ObjectStorage
from tests.support import create_test_pdf, storage_resources


class UploadsIntegrationTests(unittest.TestCase):
    """Exercise the upload router and presigned PUTs against real RustFS."""

    @classmethod
    def setUpClass(cls):
        cls.config, cls.s3 = cls.enterClassContext(storage_resources())

    def setUp(self):
        storage = ObjectStorage(config=self.config)
        self.addCleanup(storage.close)
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_storage] = lambda: storage
        self.client = self.enterContext(TestClient(app))

    def start(self):
        response = self.client.post("/uploads", json={"filename": "book.pdf"})
        self.assertEqual(response.status_code, 201, response.text)
        upload = response.json()
        self.addCleanup(self.cleanup_upload, upload)
        return upload

    def cleanup_upload(self, upload):
        try:
            self.s3.abort_multipart_upload(
                Bucket=self.config.s3_bucket_name,
                Key=upload["object_key"],
                UploadId=upload["upload_id"],
            )
        except ClientError as error:
            if error.response["Error"]["Code"] != "NoSuchUpload":
                raise
        self.s3.delete_object(
            Bucket=self.config.s3_bucket_name, Key=upload["object_key"]
        )

    def test_upload_parts_and_complete(self):
        upload = self.start()
        path = f"/uploads/{upload['upload_id']}"
        key = upload["object_key"]
        response = self.client.post(
            f"{path}/parts", json={"object_key": key, "part_numbers": [1, 2]}
        )
        self.assertEqual(response.status_code, 200, response.text)
        # S3 requires every part except the last to be at least 5 MiB.
        pdf = create_test_pdf()
        contents = [pdf + b" " * (5 * 1024 * 1024 - len(pdf)), b"\n"]
        parts = []
        with httpx.Client(timeout=15, trust_env=False) as client:
            for signed, content in zip(response.json(), contents, strict=True):
                result = client.put(signed["url"], content=content)
                self.assertEqual(result.status_code, 200, result.text)
                parts.append(
                    {
                        "part_number": signed["part_number"],
                        "etag": result.headers["etag"],
                    }
                )
        response = self.client.post(
            f"{path}/complete", json={"object_key": key, "parts": list(reversed(parts))}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["object_key"], key)
        stored = self.s3.get_object(Bucket=self.config.s3_bucket_name, Key=key)
        self.assertEqual(stored["ContentType"], "application/pdf")
        with stored["Body"] as body:
            self.assertEqual(body.read(), b"".join(contents))

    def test_validation_and_abort(self):
        self.assertEqual(
            self.client.post("/uploads", json={"filename": "book.txt"}).status_code, 422
        )
        upload = self.start()
        path = f"/uploads/{upload['upload_id']}"
        key = upload["object_key"]
        for numbers in ([0], [10001], [1, 1], []):
            self.assertEqual(
                self.client.post(
                    f"{path}/parts", json={"object_key": key, "part_numbers": numbers}
                ).status_code,
                422,
            )
        self.assertEqual(
            self.client.post(
                f"{path}/complete",
                json={
                    "object_key": key,
                    "parts": [{"part_number": 1, "etag": "missing"}],
                },
            ).status_code,
            400,
        )
        response = self.client.delete(path, params={"object_key": key})
        self.assertEqual(response.status_code, 204, response.text)
        self.assertEqual(
            self.client.delete(path, params={"object_key": key}).status_code, 404
        )
