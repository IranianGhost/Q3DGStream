FROM public.ecr.aws/docker/library/python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 openssl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir aioquic==1.7.2 pytest==9.2.0 numpy==2.5.4
WORKDIR /project
