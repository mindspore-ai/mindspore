# 构建与运行

- 环境要求
    - 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
    - 编译依赖：
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
        - [Android_NDK](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip) >= r20
    - [Git](https://git-scm.com/downloads) >= 2.28.0

- 编译构建

  在`mindspore/lite/examples/runtime_cpp`目录下执行build脚本，将能够自动下载相关文件并编译Demo。

  ```bash
  bash build.sh
  ```

  > 若MindSpore Lite推理框架下载失败，请手动下载硬件平台为CPU，操作系统为Ubuntu-x64的[MindSpore Lite 模型推理框架](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/downloads.html)，解压后将其拷贝对应到`mindspore/lite/examples/runtime_cpp/lib`目录。
  >
  > 若mobilenetv2模型下载失败，请手动下载相关模型文件[mobilenetv2](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2.ms)，并将其拷贝到`mindspore/lite/examples/runtime_cpp/model`目录。

- 文件传输

  使用`adb`将`mindspore/lite/examples/runtime_cpp\output`目录下的`runtime_cpp_demo.tar.gz`压缩包发送到Android手机

  ```shell
  adb push runtime_cpp_demo.tar.gz /data/local/tmp
  ```

- 执行推理

  使用`adb`进入Android Shell命令模式

  ```shell
  adb shell
  ```

  进入压缩包所在的相关目录，并进行解压

  ```shell
  cd /data/local/tmp && tar xzvf runtime_cpp_demo.tar.gz
  ```

  配置`LD_LIBRARY_PATH`环境变量

  ```shell
  export LD_LIBRARY_PATH=/data/local/tmp/runtime_cpp_demo/lib:{LD_LIBRARY_PATH}
  ```

  运行示例需要传递两个参数，第一个参数是模型路径，第二个参数是Option，不同的Option将会运行不同的推理流程。

  | option | 流程                        |
  | ------ | --------------------------- |
  | 0      | 基本推理流程                |
  | 1      | 输入维度Resize流程          |
  | 2      | CreateSession简化版接口流程 |
  | 3      | Session并行流程             |
  | 4      | 共享内存池流程              |
  | 5      | 回调运行流程                |

  例如：可以执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  cd ./runtime_cpp_demo/bin && ./runtime_cpp ../model/mobilenetv2.ms 0
  ```
