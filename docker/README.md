## MindSpore Dockerfile Repository

This folder hosts all the `Dockerfile` to build MindSpore container images with various hardware platforms.

### MindSpore docker build command

* CPU

    ```
    cd mindspore-cpu && docker build . -t mindspore/mindspore-cpu:0.1.0-alpha
    ```

* GPU (CUDA 9.2)

    ```
    cd mindspore-cuda9.2 && docker build . -t mindspore/mindspore-cuda9.2:0.1.0-alpha
    ```

* GPU (CUDA 10.1)

    ```
    cd mindspore-cuda10.1 && docker build . -t mindspore/mindspore-cuda10.1:0.1.0-alpha
    ```
