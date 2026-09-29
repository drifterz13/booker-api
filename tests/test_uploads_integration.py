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
                result = client.put(
                    signed["url"],
                    content=content,
                    headers={"Origin": "http://localhost:5173"},
                )
                self.assertEqual(result.status_code, 200, result.text)
                self.assertIn(
                    result.headers["access-control-allow-origin"],
                    ("*", "http://localhost:5173"),
                )
                self.assertIn(
                    "etag", result.headers["access-control-expose-headers"].lower()
                )
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

    def test_abort_upload(self):
        upload = self.start()
        path = f"/uploads/{upload['upload_id']}"
        params = {"object_key": upload["object_key"]}
        self.assertEqual(self.client.delete(path, params=params).status_code, 204)
        self.assertEqual(self.client.delete(path, params=params).status_code, 404)
