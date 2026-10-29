# MindSpore

> English | [中文](./OVERVIEW.zh.md)

## Quick Reference

- MindSpore is maintained by the [MindSpore Community](https://atomgit.com/mindspore/mindspore)

- Where to get help

    - [MindSpore Documentation](https://www.mindspore.cn/en)
    - [MindSpore Community](https://discuss.mindspore.cn)
    - [Issue Feedback](https://atomgit.com/mindspore/mindspore/issues)

## MindSpore

MindSpore is a new open source deep learning training/inference framework that could be used for mobile, edge and cloud scenarios. MindSpore is designed to provide development experience with friendly design and efficient execution for the data scientists and algorithmic engineers, native support for Atlas AI processor, and software hardware co-optimization. At the meantime MindSpore as a global AI open source community, aims to further advance the development and enrichment of the AI software/hardware application ecosystem.

## Supported Tags and Dockerfile Usage

### Tag Specification

Tags follow this format:

```text
<MindSpore Version>-<CANN Version>-<Hardware Info (Chip)>-<Operating System>-<Python Version>
```

| Field | Example Values | Description |
|-------|---------------|-------------|
| MindSpore Version | 2.11.0 | Corresponds to the version identifier in MindSpore official release tags |
| CANN Version | cann9.2.0 | Corresponds to the version identifier in Atlas official release tags |
| Hardware Info (Chip) | see below | Atlas chip model identifier |
| Operating System | ubuntu22.04 / openeuler24.03 | Operating system distribution and version used in the base image |
| Python Version | py3.12 | Major Python version built into the image |

> Tips: System architecture is automatically detected via Docker Manifest, no need to specify in the tag.

### Tag

- `2.11.0-cann9.2.0-910b-ubuntu22.04-py3.12`
- `2.11.0-cann9.2.0-910b-openeuler24.03-py3.12`
- `2.11.0-cann9.2.0-a3-ubuntu22.04-py3.12`
- `2.11.0-cann9.2.0-a3-openeuler24.03-py3.12`

See dockerfile:[dockerfile](https://atomgit.com/mindspore/mindspore/blob/master/docker/Dockerfile)

### Building Arguments

| Parameter | Description | Required | Source | Example Values |
|-----------|-------------|----------|--------|----------------|
| CANN_VERSION | Atlas CANN toolkit version | Yes | CANN image tag | 9.2.0 |
| CHIP_ARCH | Atlas chip architecture identifier | Yes | Tag specification | see below |
| OS_SYSTEM | Base image operating system and version | Yes | Tag specification | ubuntu22.04 / openeuler24.03 |
| PY_VERSION | Python version built into the base image | Yes | Tag specification | py3.12 |
| MINDSPORE_VERSION | MindSpore version number | Yes | [MindSpore repository releases](https://atomgit.com/mindspore/mindspore/releases) | 2.11.0 |
| PIP_INDEX_URL | pip installation source URL (default: Huawei Cloud mirror) | No | PyPI mirror source | https://mirrors.huaweicloud.com/repository/pypi/simple |

### Image Repository Address

MindSpore Atlas images are hosted on Huawei Cloud SWR image repository:

```text
swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore
```

**Full Image Example:**

```text
swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.11.0-cann9.2.0-910b-ubuntu22.04-py3.12
```

## Quick Start

### Building MindSpore Image

Taking the construction of the 2.11.0-cann9.2.0-910b-ubuntu22.04-py3.12 image as an example:

```bash
docker build \
    --build-arg CANN_VERSION=9.2.0 \
    --build-arg CHIP_ARCH=910b \
    --build-arg OS_SYSTEM=ubuntu22.04 \
    --build-arg PY_VERSION=py3.12 \
    --build-arg MINDSPORE_VERSION=2.11.0 \
    --build-arg PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
    -t mindspore:2.11.0-cann9.2.0-910b-ubuntu22.04-py3.12 \
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
    -it mindspore:{tag} bash
```

Where:

- `{tag}` is the tag specified during the MindSpore image build, for example, `mindspore:2.11.0-cann9.2.0-910b-ubuntu22.04-py3.12`.

### How to Extend for Custom Development

```bash
# Use MindSpore image as base image, add user software
FROM swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.11.0-cann9.2.0-910b-ubuntu22.04-py3.12

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

## Image Usage Responsibility

The released Atlas software images are community versions, not intended for commercial use, and serve only as references for production practices.