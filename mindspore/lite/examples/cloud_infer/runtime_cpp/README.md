# 构建与运行

- 环境要求
    - 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
    - 编译依赖：
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- 编译构建

  在`mindspore/lite/examples/cloud_infer/runtime_cpp/`目录下执行build脚本，将能够自动下载相关文件并编译Demo。

  ```bash
  bash build.sh
  ```

  > 若MindSpore Lite推理框架下载失败，请手动下载硬件平台为CPU，操作系统为Ubuntu-x64的[MindSpore Lite 模型推理框架](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)
  ，将压缩包拷贝到`mindspore/lite/examples/cloud_infer/runtime_cpp/`目录。
  >
  > 若mobilenetv2模型下载失败，请手动下载相关模型文件[mobilenetv2](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2.mindir)
  ，并将其拷贝到`mindspore/lite/examples/cloud_infer/runtime_cpp/model`目录。

- 执行推理

  设置环境变量`LITE_HOME`为MindSpore Lite tar包解压路径，并设置环境变量`LD_LIBRARY_PATH`：

  ```bash
  export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
  ```

  可以执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  cd build && ./runtime_cpp --model_path=../model/mobilenetv2.mindir --device_type=CPU
  ```
