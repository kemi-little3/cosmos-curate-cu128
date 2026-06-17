FROM cosmos-curate:slim

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libglx0 \
        libglvnd0 \
        libx11-6 \
        libxext6 \
        libxcb1 \
        libxau6 \
    && rm -rf /var/lib/apt/lists/*

