# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ObjectStorageSDK-backed storage client for Data Engine object storage."""

from __future__ import annotations

import mimetypes
import os
import pathlib
from typing import TYPE_CHECKING, Any

import attrs
from loguru import logger

from cosmos_curate.core.utils.storage.s3_client import S3Prefix, is_s3path
from cosmos_curate.core.utils.storage.storage_client import (
    DOWNLOAD_CHUNK_SIZE_BYTES,
    UPLOAD_CHUNK_SIZE_BYTES,
    BackgroundUploader,
    BaseClientConfig,
    StorageClient,
    StoragePrefix,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


DEFAULT_SDK_BUCKETS = "data-lake"
DEFAULT_ENDPOINT = "127.0.0.1:9000"
DEFAULT_ACCESS_KEY = "admin"
DEFAULT_SECRET_KEY = "zjulearning123"  # noqa: S105


@attrs.define
class ObjectStorageSDKClientConfig(BaseClientConfig):
    """Configuration for the Data Engine object storage SDK client."""

    endpoint: str = attrs.field(default=DEFAULT_ENDPOINT)
    access_key: str = attrs.field(default=DEFAULT_ACCESS_KEY)
    secret_key: str = attrs.field(default=DEFAULT_SECRET_KEY)
    secure: bool = attrs.field(default=False)
    bucket: str = attrs.field(default="data-lake")
    region: str | None = attrs.field(default=None)


def _env_any(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_endpoint(endpoint: str, secure: bool) -> tuple[str, bool]:
    stripped = endpoint.strip()
    if stripped.startswith("http://"):
        return stripped.removeprefix("http://"), False
    if stripped.startswith("https://"):
        return stripped.removeprefix("https://"), True
    return stripped, secure


def _configured_buckets() -> set[str]:
    raw = _env_any(
        "COSMOS_OBJECT_STORAGE_SDK_BUCKETS",
        "OBJECT_STORAGE_SDK_BUCKETS",
        default=DEFAULT_SDK_BUCKETS,
    )
    if raw is None:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def should_use_object_storage_sdk(target_path: str | None) -> bool:
    """Return True when an s3:// path should use ObjectStorageSDK instead of boto3."""

    if not is_s3path(target_path):
        return False
    buckets = _configured_buckets()
    if not buckets:
        return False
    assert target_path is not None
    bucket = S3Prefix(target_path).bucket
    return "*" in buckets or bucket in buckets


def _content_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def get_object_storage_sdk_client_config(
    target_path: str,
    *,
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> ObjectStorageSDKClientConfig:
    """Create ObjectStorageSDK client config from environment with Data Engine defaults."""

    bucket = S3Prefix(target_path).bucket
    secure = _parse_bool(
        _env_any("COSMOS_OBJECT_STORAGE_SDK_SECURE", "OBJECT_STORAGE_SDK_SECURE"),
        default=False,
    )
    endpoint, secure = _normalize_endpoint(
        _env_any(
            "COSMOS_OBJECT_STORAGE_SDK_ENDPOINT",
            "OBJECT_STORAGE_SDK_ENDPOINT",
            default=DEFAULT_ENDPOINT,
        )
        or DEFAULT_ENDPOINT,
        secure,
    )
    return ObjectStorageSDKClientConfig(
        endpoint=endpoint,
        access_key=_env_any(
            "COSMOS_OBJECT_STORAGE_SDK_ACCESS_KEY",
            "OBJECT_STORAGE_SDK_ACCESS_KEY",
            default=DEFAULT_ACCESS_KEY,
        )
        or DEFAULT_ACCESS_KEY,
        secret_key=_env_any(
            "COSMOS_OBJECT_STORAGE_SDK_SECRET_KEY",
            "OBJECT_STORAGE_SDK_SECRET_KEY",
            default=DEFAULT_SECRET_KEY,
        )
        or DEFAULT_SECRET_KEY,
        secure=secure,
        bucket=bucket,
        region=_env_any("COSMOS_OBJECT_STORAGE_SDK_REGION", "OBJECT_STORAGE_SDK_REGION"),
        can_overwrite=can_overwrite,
        can_delete=can_delete,
    )


class ObjectStorageSDKClient(StorageClient):
    """StorageClient implementation backed by Data Engine ObjectStorageSDK."""

    def __init__(self, config: ObjectStorageSDKClientConfig) -> None:
        try:
            from object_storage_sdk import ObjectStorageConfig, ObjectStorageSDK
        except ModuleNotFoundError as exc:
            msg = (
                "object_storage_sdk is required only when writing to configured "
                "Data Engine object-storage buckets."
            )
            raise RuntimeError(msg) from exc

        self._config = config
        self._sdk = ObjectStorageSDK(
            ObjectStorageConfig(
                endpoint=config.endpoint,
                access_key=config.access_key,
                secret_key=config.secret_key,
                secure=config.secure,
                bucket=config.bucket,
                region=config.region,
            )
        )
        self.can_overwrite = config.can_overwrite
        self.can_delete = config.can_delete

    def object_exists(self, dest: StoragePrefix) -> bool:
        assert isinstance(dest, S3Prefix)
        if not dest.prefix:
            return self._sdk.bucket_exists(dest.bucket)
        return self._sdk.object_exists(dest.prefix, bucket=dest.bucket)

    def upload_bytes(self, dest: StoragePrefix, data: "bytes | npt.NDArray[np.uint8]") -> None:
        assert isinstance(dest, S3Prefix)
        if not self.can_overwrite and self.object_exists(dest):
            error_msg = f"Object {dest.path} already exists and overwriting is not allowed."
            raise ValueError(error_msg)
        payload = data if isinstance(data, bytes) else memoryview(data).tobytes()
        self._sdk.put_bytes(dest.prefix, payload, bucket=dest.bucket, content_type=_content_type(dest.prefix))

    def upload_bytes_uri(
        self,
        uri: str,
        data: bytes,
        chunk_size_bytes: int = UPLOAD_CHUNK_SIZE_BYTES,  # noqa: ARG002
    ) -> None:
        self.upload_bytes(S3Prefix(uri), data)

    def download_object_as_bytes(
        self,
        uri: StoragePrefix,
        chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES,  # noqa: ARG002
    ) -> bytes:
        assert isinstance(uri, S3Prefix)
        return self._sdk.get_object_bytes(uri.prefix, bucket=uri.bucket)

    def download_objects_as_bytes(self, uris: list[StoragePrefix]) -> list[bytes]:
        return [self.download_object_as_bytes(uri) for uri in uris]

    def list_recursive_directory(self, uri: StoragePrefix, limit: int = 0) -> list[StoragePrefix]:
        assert isinstance(uri, S3Prefix)
        return [S3Prefix(f"s3://{uri.bucket}/{item['Key']}") for item in self.list_recursive(uri, limit)]

    def list_recursive(self, prefix: StoragePrefix, limit: int = 0) -> list[dict[str, Any]]:
        assert isinstance(prefix, S3Prefix)
        self._sdk.ensure_bucket(prefix.bucket)
        keys = self._sdk.list_objects(prefix.prefix, bucket=prefix.bucket, recursive=True)
        if limit > 0:
            keys = keys[:limit]
        return [{"Key": key} for key in keys]

    def upload_file(
        self,
        local_path: str,
        remote_path: StoragePrefix,
        chunk_size: int = UPLOAD_CHUNK_SIZE_BYTES,  # noqa: ARG002
    ) -> None:
        assert isinstance(remote_path, S3Prefix)
        if not self.can_overwrite and self.object_exists(remote_path):
            error_msg = f"Object {remote_path.path} already exists and overwriting is not allowed."
            raise ValueError(error_msg)

        logger.info(f"Uploading {local_path} to {remote_path} via ObjectStorageSDK")
        self._sdk.upload_file(
            local_path,
            remote_path.prefix,
            bucket=remote_path.bucket,
            content_type=_content_type(remote_path.prefix),
        )
        logger.info(f"Upload complete: {remote_path}")

    def sync_remote_to_local(
        self,
        remote_prefix: StoragePrefix,
        local_dir: pathlib.Path,
        *,
        delete: bool = False,
        chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES,  # noqa: ARG002
    ) -> None:
        assert isinstance(remote_prefix, S3Prefix)
        local_dir.mkdir(parents=True, exist_ok=True)
        objects = self.list_recursive(remote_prefix)
        remote_prefix_path = remote_prefix.prefix.rstrip("/")
        seen: set[pathlib.Path] = set()
        for obj in objects:
            key = str(obj["Key"])
            relative = key[len(remote_prefix_path) :].lstrip("/") if remote_prefix_path else key
            target = local_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            self._sdk.download_file(key, target, bucket=remote_prefix.bucket)
            seen.add(target)
        if delete:
            for local_file in local_dir.rglob("*"):
                if local_file.is_file() and local_file not in seen:
                    local_file.unlink()

    def make_background_uploader(
        self,
        chunk_size_bytes: int = UPLOAD_CHUNK_SIZE_BYTES,
    ) -> "ObjectStorageSDKBackgroundUploader":
        return ObjectStorageSDKBackgroundUploader(self, chunk_size_bytes)

    def delete_object(self, dest: StoragePrefix) -> None:
        assert isinstance(dest, S3Prefix)
        if not self.can_delete:
            error_msg = f"Deletion not allowed for object {dest.prefix} in bucket {dest.bucket}."
            raise ValueError(error_msg)
        if not dest.prefix:
            error_msg = f"Refusing to delete bucket root {dest.path}."
            raise ValueError(error_msg)
        self._sdk.remove_object(dest.prefix, bucket=dest.bucket)


class ObjectStorageSDKBackgroundUploader(BackgroundUploader):
    """Background uploader backed by ObjectStorageSDK."""

    def __init__(self, client: ObjectStorageSDKClient, chunk_size_bytes: int) -> None:
        super().__init__(client, chunk_size_bytes)

    def add_task_file(self, local_path: pathlib.Path, remote_path: str) -> None:
        future = self.executor.submit(self._upload_file, local_path, remote_path)
        self.futures.append(future)

    def _upload_file(self, local_path: pathlib.Path, remote_path: str) -> None:
        remote_prefix = S3Prefix(remote_path)
        self.client.upload_file(str(local_path), remote_prefix, self.chunk_size_bytes)  # type: ignore[attr-defined]


def create_object_storage_sdk_client(
    target_path: str | None = None,
    profile_name: str = "default",  # noqa: ARG001
    *,
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> ObjectStorageSDKClient | None:
    """Create an ObjectStorageSDK client for configured s3:// buckets."""

    if not should_use_object_storage_sdk(target_path):
        return None
    assert target_path is not None
    return ObjectStorageSDKClient(
        get_object_storage_sdk_client_config(
            target_path,
            can_overwrite=can_overwrite,
            can_delete=can_delete,
        )
    )
