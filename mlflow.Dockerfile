FROM python:3.12-slim

# Update all packages to their latest versions to minimize vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Bangkok

RUN apt update && apt upgrade -y \
    && apt install -y tzdata locales curl \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en

RUN pip3 install --upgrade pip
RUN pip3 install mlflow

# Set working directory and volumes
WORKDIR /mlflow
VOLUME ["/mlflow"]
EXPOSE 5000
# Ensure MLflow server runs correctly on container start
CMD mlflow server -h 0.0.0.0 -w 2
