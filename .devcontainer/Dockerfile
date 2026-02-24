#FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
#FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ---------------------------
# Add developer user with passwordless sudo
# ---------------------------
ARG USER=developer
RUN apt-get update && apt-get install -y sudo \
    && useradd -ms /bin/bash ${USER} \
    && usermod -aG sudo ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /workspace

# ---------------------------
# Install tzdata and other system dependencies as root
# ---------------------------
USER root
RUN apt-get update && \
    apt-get install -y tzdata lsb-release gnupg2 curl build-essential git cmake python3-pip && \
    ln -fs /usr/share/zoneinfo/Asia/Kolkata /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*
# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev python3-distutils \
    curl gnupg2 lsb-release wget nano tree bash-completion && \
    rm -rf /var/lib/apt/lists/*

# ROS setup
RUN mkdir -p /usr/share/keyrings && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /tmp/ros.key && \
    gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg /tmp/ros.key && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    | tee /etc/apt/sources.list.d/ros-latest.list > /dev/null && \
    rm /tmp/ros.key && \
    apt-get update && apt-get install -y ros-noetic-desktop-full python3-catkin-tools && \
    rm -rf /var/lib/apt/lists/*

# TensorRT install
# RUN wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub && \
#     apt-key add 7fa2af80.pub && rm 7fa2af80.pub && \
#     echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" \
#       > /etc/apt/sources.list.d/tensorrt.list && \
#     apt-get update && \
#     apt-get install -y libnvinfer8 libnvinfer-plugin8 libnvinfer-dev libnvinfer-plugin-dev python3-libnvinfer python3-libnvinfer-dev && \
#     rm -rf /var/lib/apt/lists/*

# ---------------------------
# Install Python 3.10
# ---------------------------
USER root
RUN apt-get update && apt-get install -y \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libffi-dev llvm libncurses5-dev libncursesw5-dev libomp5 \
    xz-utils tk-dev git curl && rm -rf /var/lib/apt/lists/*

USER ${USER}
# ENV PYENV_ROOT="/home/${USER}/.pyenv"
# ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

# RUN curl https://pyenv.run | bash && \
#     /bin/bash -lc "pyenv install 3.10.12 && pyenv global 3.10.12" && \
#     /bin/bash -lc "python3 --version && pip install --upgrade pip setuptools wheel"

# # Add pyenv init to shell
# RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
#     echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
#     echo 'eval "$(pyenv init --path)"' >> ~/.bashrc && \
#     echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# ---------------------------
# Switch to developer user
# ---------------------------
USER ${USER}
# Optional: install cuML using the new python
#RUN /bin/bash -lc "pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com cuml-cu11"

# ---------------------------
# Source ROS automatically
# ---------------------------
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
