# Android编译部署

 `Linux` `Android`  `IOT` `C/C++` `全流程` `模型编译` `模型代码生成` `模型部署` `推理应用` `初级` `中级` `高级`

<!-- TOC -->

- Android编译部署
    - [编译依赖](#编译依赖)
    - [工程构建](#工程构建)
    - [工程部署](#工程部署)
    - [更多详情](#更多详情)
        - [Linux_x86_64编译部署](#Linux_x86_64编译部署)
        - [STM32F746编译部署](#STM32F746编译部署)

<!-- /TOC -->

## Android编译部署

本教程以MobileNetv2在安卓手机编译部署为例，使用用户快速了解codegen在安卓平台生成代码、工程构建以及部署的一系列流程。关于converter、codegen的获取以及详细参数介绍可参考mindspore的[编译构建介绍](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html)。

### 编译依赖

安卓平台的编译部署需要提前配置ANDROID_NDK到环境变量。

- NDK 21.3
- [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
- [CMake](https://cmake.org/download/) >= 3.18.3

### 工程构建

#### 快速使用

进入`mindspore/mindspore/lite/micro/examples/mobilenetv2`目录执行脚本`mobilenetv2.sh`自动生成模型推理代码并编译工程目录

```
bash mobilenetv2.sh
```

codegen编译[MobileNetv2模型](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_imagenet/r1.2/mobilenetv2.ms)，生成对应的模型推理代码。具体命令如下：

```bash
./codegen --codePath=. --modelPath=mobilenetv2.ms --target=ARM64
```

关于codegen的更多使用命令说明，可参见[codegen工具的详细介绍](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/downloads.html)

#### 生成代码工程说明

```bash
├── mobilenetv2
└── operator_library
```

##### 算子静态库目录说明

在编译此工程之前需要预先获取安卓平台对应的[Release包](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/downloads.html)。

安卓平台对应的Release包的目录如下:
```text
mindspore-lite-{version}-inference-android-{arch}
├── inference
│   ├── include     # 推理框架头文件
│   ├── lib         # 推理框架库
│   │   ├── libmindspore-lite.a  # MindSpore Lite推理框架的静态库
│   │   └── libmindspore-lite.so # MindSpore Lite推理框架的动态库
│   ├── minddata    # 图像处理库
│   │   ├── include
│   │   └── lib
│   │       └── libminddata-lite.so # 图像处理动态库文件
│   └── third_party # NPU库
│       └── hiai_ddk
└── tools
    ├── benchmark # 基准测试工具
    │   └── benchmark
    └── codegen   # 代码生成工具
        ├── include  # 算子头文件
        └── lib      # 算子静态库
```

生成代码工程目录如下：

```bash
├── mobilenetv2         # 生成代码的根目录
    ├── benchmark       # 生成代码的benchmark目录
    └── src             # 模型推理代码目录
```

#### 代码工程编译

组织生成的模型推理代码以及安卓平台算子静态库编译模型推理静态库

进入代码工程目录，新建并进入build目录

```bash
mkdir mobilenetv2/build && cd mobilenetv2/build
```

开始编译

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang" \
-DANDROID_NATIVE_API_LEVEL="19" \
-DPLATFORM_ARM64=ON \
-DPKG_PATH={path to}/mindspore-lite-{version}-inference-android-{arch} ..
make
```

`{path to}`和`{version}`需要用户根据实际情况填写。若用户需要编译安卓arm32环境，则使用:

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
-DANDROID_ABI="armeabi-v7a" \
-DANDROID_TOOLCHAIN_NAME="clang" \
-DANDROID_NATIVE_API_LEVEL="19" \
-DPLATFORM_ARM32=ON \
-DPKG_PATH={path to}/mindspore-lite-{version}-inference-android-{arch} ..
make
```

此时在`mobilenetv2/build/src/`目录下生成了`libnet.a`，推理执行库，在`mobilenetv2/build`目录下生成了`benchmark`可执行文件。而对应的模型参数文件net.bin在生成的代码src目录下。

### 工程部署

adb将生成的可执行二进制文件benchmark、mobilenetv2_input.bin以及模型参数文件net.bin拷贝到目标安卓服务器，执行以下命令即可：

```bash
./benchmark mobilenetv2_input.bin net.bin 100
```

#### 执行结果

```bash
=========run benchmark========
input 0: mobilenetv2_input.bin
name: Softmax-65, ,DataType: 43, Size: 4004, Shape:1 1001, Data:
0.000010,0.000010,0.000014,0.000091,0.000080,0.000717,0.000112,0.000738,0.000008,0.000003
=========run success========
```

## 更多详情

### [Linux_x86_64编译部署](https://www.mindspore.cn/tutorial/lite/zh-CN/master/quick_start/quick_start_codegen.html)

### [STM32F746编译部署](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/micro/example/mnist_stm32f746)

