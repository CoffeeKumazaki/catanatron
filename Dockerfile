# This is a Container we can use to run Experiments in Paperspace.
#   Edit, build, and publish to DockerHub.
#   `docker build . -f Dockerfile.paperspace -t bcollazo/paperspace-rl` to build.
#   `docker run --rm -it bcollazo/paperspace-rl catanatron-play` to ensure it works.
#   `docker push bcollazo/paperspace-rl` to publish.
# FROM paperspace/tensorflow:2.0.0-gpu-py3-jupyter-lab
FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

# Install Python3.8
RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# We copy just the dependencies first to leverage Docker cache
COPY dev-requirements.txt .

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install -r dev-requirements.txt

EXPOSE 8888