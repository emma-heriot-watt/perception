ARG IMAGE_BASE_NAME

# hadolint ignore=DL3006
FROM ${IMAGE_BASE_NAME}

# Install necessary dependencies for OpenCV
RUN apt-get update -qq \
	&& apt-get upgrade -y --no-install-recommends \
	ffmpeg libsm6 libxext6 \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get autoremove -y

ARG TORCH_VERSION_SUFFIX=""

WORKDIR ${PYSETUP_PATH}/repo

COPY . ${PYSETUP_PATH}/repo

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN poetry install --only main \
	&& TORCH_VERSION="$(pip show torch | grep Version | cut -d ':' -f2 | xargs)${TORCH_VERSION_SUFFIX}" \
	&& TORCHVISION_VERSION="$(pip show torchvision | grep Version | cut -d ':' -f2 | xargs)${TORCH_VERSION_SUFFIX}" \
	&& pip install --no-cache-dir torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" -f https://download.pytorch.org/whl/torch_stable.html

# Set the PYTHONPATH
ENV PYTHONPATH='./src'

ENTRYPOINT ["/bin/bash"]
