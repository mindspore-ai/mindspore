## MindSpore Dockerfile Repository

This folder hosts all the `Dockerfile` to build MindSpore container images with various hardware platforms.

### MindSpore docker build command

| Hardware Platform | Version | Build Command |
| :---------------- | :------ | :------------ |
| CPU | `x.y.z` | cd mindspore-cpu/x.y.z && docker build . -t mindspore/mindspore-cpu:x.y.z |
|  | `devel` | cd mindspore-cpu/devel && docker build . -t mindspore/mindspore-cpu:devel |
|  | `runtime` | cd mindspore-cpu/runtime && docker build . -t mindspore/mindspore-cpu:runtime |
| GPU | `x.y.z` | cd mindspore-gpu/x.y.z  && docker build . -t mindspore/mindspore-gpu:x.y.z  |
|  | `devel` | cd mindspore-gpu/devel && docker build . -t mindspore/mindspore-gpu:devel |
|  | `runtime` | cd mindspore-gpu/runtime && docker build . -t mindspore/mindspore-gpu:runtime |

> **NOTICE:** The `x.y.z` version shown above should be replaced with the real version number.
