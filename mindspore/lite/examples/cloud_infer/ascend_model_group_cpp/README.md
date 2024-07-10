# 构建与运行

- 环境要求
    - 系统环境：Linux
    - 编译依赖：
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- 编译构建

  > 根据系统环境，请手动下载x86_64或aarch64平台的的[MindSpore Lite 云侧推理包](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)
  ，并解压。
  >
  > 请手动下载相关模型文件[mobilenetv2](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir)
  ，并将其拷贝到`mindspore/lite/examples/cloud_infer/quick_start_cpp/`目录。

  设置环境变量`LITE_HOME`为MindSpore Lite tar包解压路径，并设置环境变量`LD_LIBRARY_PATH`：

  ```bash
  export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/runtime/third_party/dnnl:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
  ```

  编译构建

  ```bash
  rm -rf build
  mkdir build && cd build
  cmake ../ && make
  ```

- 执行推理

  可以执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  ./mindspore_quick_start_cpp ../mobilenetv2.mindir
  ```