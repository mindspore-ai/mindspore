## MindSpore Dockerfile Repository

This folder hosts all the `Dockerfile` to build MindSpore container images with various hardware platforms.

### MindSpore docker build command

* CPU

    ```
    cd mindspore-cpu/0.1.0-alpha && docker build . -t mindspore/mindspore-cpu:0.1.0-alpha
    ```

* GPU

    ```
    cd mindspore-gpu/0.1.0-alpha && docker build . -t mindspore/mindspore-gpu:0.1.0-alpha
    ```
