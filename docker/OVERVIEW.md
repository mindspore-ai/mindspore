# MindSpore

> English | [中文](./OVERVIEW.zh.md)

## Quick Reference

- MindSpore is maintained by the [MindSpore Community](https://atomgit.com/mindspore/mindspore)

- Where to get help

    - [MindSpore Documentation](https://www.mindspore.cn/en)
    - [MindSpore Community](https://discuss.mindspore.cn)
    - [Issue Feedback](https://atomgit.com/mindspore/mindspore/issues)

## MindSpore

MindSpore is a new open source deep learning training/inference framework that could be used for mobile, edge and cloud scenarios. MindSpore is designed to provide development experience with friendly design and efficient execution for the data scientists and algorithmic engineers, native support for Ascend AI processor, and software hardware co-optimization. At the meantime MindSpore as a global AI open source community, aims to further advance the development and enrichment of the AI software/hardware application ecosystem.

## Supported Tags and Dockerfile Usage

### Tag Specification

Tags follow this format:

```text
<MindSpore Version>-<Hardware Info (Chip)>-<Operating System>-<Python Version>
```

| Field | Example Values | Description |
|-------|---------------|-------------|
| MindSpore Version | 2.9.0 | Corresponds to the version identifier in MindSpore official release tags |
| Hardware Info (Chip) | see below | Ascend chip model identifier |
| Operating System | ubuntu22.04 / openeuler24.03 | Operating system distribution and version used in the base image |
| Python Version | py3.11 | Major Python version built into the image |

> Tips: System architecture is automatically detected via Docker Manifest, no need to specify in the tag.

### Image Repository Address

MindSpore Ascend images are hosted on Huawei Cloud SWR image repository:

```text
swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore
```

**Full Image Example:**

```text
swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.9.0-910b-ubuntu22.04-py3.11
```

### Building Arguments

| Parameter | Description | Required | Source | Example Values |
|-----------|-------------|----------|--------|----------------|
| CANN_VERSION | Ascend CANN toolkit version | Yes | CANN image tag | 9.0.0 |
| CHIP_ARCH | Ascend chip architecture identifier | Yes | Tag specification | see below |
| OS_SYSTEM | Base image operating system and version | Yes | Tag specification | ubuntu22.04 / openeuler24.03 |
| PY_VERSION | Python version built into the base image | Yes | Tag specification | py3.11 |
| MINDSPORE_VERSION | MindSpore version number | Yes | MindSpore repository releases | 2.9.0 |
| PIP_INDEX_URL | pip installation source URL (default: Huawei Cloud mirror) | No | PyPI mirror source | https://mirrors.huaweicloud.com/repository/pypi/simple |

## Quick Start

### Building MindSpore Image

```bash
docker build \
--build-arg CANN_VERSION=9.0.0 \
--build-arg CHIP_ARCH=910b \
--build-arg OS_SYSTEM=ubuntu22.04 \
--build-arg PY_VERSION=py3.11 \
--build-arg MINDSPORE_VERSION=2.9.0 \
--build-arg PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
-t mindspore:2.9.0-910b-ubuntu22.04-py3.11 \
-f Dockerfile .
```

### Running MindSpore Container

```bash
docker run \
    --privileged \
    --name mindspore_container \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it mindspore:tag bash
```

### How to Extend for Custom Development

```bash
# Use MindSpore image as base image, add user software
FROM swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.9.0-910b-ubuntu22.04-py3.11

RUN apt update -y && \
    apt install -y gcc g++

# Install additional dependencies
RUN pip install pandas scikit-learn

# Copy user code
WORKDIR /workspace
COPY . /workspace

CMD ["python", "train.py"]
```

## Supported Hardware

| Chip Series | Product Examples | Architecture |
|-------------|-----------------|--------------|
| Atlas A2 | Atlas 800T A2, Atlas 900 A2 PoD | Auto-detected (ARM64/x86_64) |
| Atlas A3 | Atlas 800T A3 | Auto-detected (ARM64/x86_64) |

> Tips: Use the `docker manifest inspect` command to view the system architectures supported by an image.

## License

View the [license information](https://atomgit.com/mindspore/mindspore/blob/master/LICENSE) for MindSpore included in these images.

As with all container images, pre-installed software packages (Python, system libraries, etc.) may be subject to their respective licenses.