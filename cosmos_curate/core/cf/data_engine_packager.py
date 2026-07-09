# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pack Data Engine source URIs into a tar archive and write manifest/output."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger

from cosmos_curate.core.cf.data_engine_adapter import DataEngineRequest
from cosmos_curate.core.utils.storage.storage_utils import get_storage_client, path_to_prefix, read_bytes
from cosmos_curate.core.utils.storage.writer_utils import write_bytes, write_json


def _uri_basename(uri: str, fallback: str) -> str:
    parsed = urlparse(uri)
    name = Path(parsed.path).name
    return name or fallback


def _make_member_name(index: int, source_uri: str) -> str:
    base = _uri_basename(source_uri, f"source-{index}")
    return f"{index:06d}_{base}"


def _create_tar_bytes(files: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for filename, data in files:
            info = tarfile.TarInfo(name=filename)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer.getvalue()


def _output_object_prefix(output_uri: str) -> str:
    parsed = path_to_prefix(output_uri)
    return parsed.path.rstrip("/")


def pack_source_uris(request: DataEngineRequest) -> None:
    """Download source URIs, package them into a tar, and write manifest/output."""

    output_client = get_storage_client(request.args.output_uri, can_overwrite=True, profile_name="default")
    if output_client is None:
        msg = f"Output URI is not a remote object store path: {request.args.output_uri}"
        raise ValueError(msg)

    output_prefix = _output_object_prefix(request.args.output_uri)
    tar_object = path_to_prefix(f"{output_prefix}/batch-000001.tar")
    manifest_object = path_to_prefix(f"{output_prefix}/batch-000001.json")

    files_to_write: list[tuple[str, bytes]] = []
    manifest_files: list[dict[str, object]] = []
    for idx, source_uri in enumerate(request.args.source_uris):
        source_client = get_storage_client(source_uri, profile_name="default")
        data = read_bytes(source_uri, source_client)
        member_name = _make_member_name(idx, source_uri)
        files_to_write.append((member_name, data))
        manifest_files.append(
            {
                "source_uri": source_uri,
                "source_name": _uri_basename(source_uri, f"source-{idx}"),
                "member_name": member_name,
                "size": len(data),
            },
        )

    tar_bytes = _create_tar_bytes(files_to_write)
    manifest = {
        "tar_file": "batch-000001.tar",
        "file_count": len(manifest_files),
        "files": manifest_files,
    }

    logger.info("Writing tar output to %s", tar_object.path)
    write_bytes(
        tar_bytes,
        tar_object,
        "data-engine-tar",
        request.pipeline_task_id,
        verbose=False,
        client=output_client,
        overwrite=True,
    )
    logger.info("Writing manifest output to %s", manifest_object.path)
    write_json(
        manifest,
        manifest_object,
        "data-engine-manifest",
        request.pipeline_task_id,
        verbose=False,
        client=output_client,
        overwrite=True,
    )
