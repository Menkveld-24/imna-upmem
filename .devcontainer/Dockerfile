# Latest supported version of UPMEM SDK
FROM ubuntu:22.04

ARG UPMEM_SDK_VERSION=2025.1.0
ARG CONTAINER_TIMEZONE=Etc/UTC

RUN apt-get update

RUN apt-get install -y wget git ninja-build xxd

# Fix the timezone otherwise python will ask for it during installation
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y

# Install Python 3.13
RUN apt-get install -y python3.13 python3.13-venv python3.13-dev
# Create the aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# Fix the distutils dependecy for dpu-profiling
RUN python3 -m ensurepip --upgrade
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install the UPMEM SDK
COPY SDK /tmp/SDK
# Install SDK
RUN apt install /tmp/SDK/${UPMEM_SDK_VERSION}/upmem_${UPMEM_SDK_VERSION}_amd64.deb -y
# Only needed for V1B DIMMs
# RUN apt install /tmp/SDK/${UPMEM_SDK_VERSION}/upmem-firmware_${UPMEM_SDK_VERSION}_amd64.deb -y
# For machines with real hardware
# RUN apt install /tmp/SDK/${UPMEM_SDK_VERSION}/upmem-driver-dkms_${UPMEM_SDK_VERSION}_amd64.deb -y
RUN rm -rf /tmp/SDK

# Install Go 1.25.3
RUN wget https://go.dev/dl/go1.25.3.linux-amd64.tar.gz -O /tmp/go1.25.3.linux-amd64.tar.gz
RUN rm -rf /usr/local/go
RUN tar -C /usr/local -xzf /tmp/go1.25.3.linux-amd64.tar.gz
# Make Go available everywhere (build, runtime, interactive)
ENV PATH="/usr/local/go/bin:${PATH}"

# Install Cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v4.1.2/cmake-4.1.2-linux-x86_64.sh -O /tmp/cmake-4.1.2-linux-x86_64.sh
RUN sh /tmp/cmake-4.1.2-linux-x86_64.sh --skip-license --prefix=/usr/local
RUN rm -f /tmp/cmake-4.1.2-linux-x86_64.sh

# Make every shell source it
RUN echo "source /usr/upmem_env.sh simulator" >> /etc/bash.bashrc

# COPY .devcontainer/entrypoint.sh /usr/local/bin/entrypoint.sh
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Install cuda, see: https://developer.nvidia.com/cuda-downloads
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN rm cuda-keyring_1.1-1_all.deb
RUN apt-get -y install cuda-toolkit-13-0