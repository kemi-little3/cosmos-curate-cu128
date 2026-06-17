# Fork notice

This repository is a fork of NVIDIA's cosmos-curate
(https://github.com/nvidia-cosmos/cosmos-curate) at commit 29f1358.

Licensed under Apache License 2.0. The original LICENSE and NOTICE are preserved.

Local modifications in this fork:
- pixi.toml / pixi.lock — PyPI index switched to aliyun mirror for China network
- package/cosmos_curate/default.dockerfile.jinja2 — inject Tsinghua conda mirror at build time
- cosmos_curate/client/local_cli/launch_local.py — tolerant mkdir in pixi-path symlink preamble
