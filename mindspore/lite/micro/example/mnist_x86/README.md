# X86编译部署

 `Linux` `IoT` `C++` `全流程` `模型编译` `模型代码生成` `模型部署` `推理应用` `初级` `中级` `高级`

<!-- TOC -->

- [X86编译部署](#X86编译部署)
    - [概述](#概述)
    - [模型编译体验](#模型编译体验)
    - [详细步骤](#详细步骤)
        - [生成代码](#生成代码)
        - [部署应用](#部署应用)
            - [编译依赖](#编译依赖)
            - [构建与运行](#构建与运行)
            - [编写推理代码示例](#编写推理代码示例)
    - [更多详情](#更多详情)
        - [Android平台编译部署](#android平台编译部署)
        - [Arm&nbsp;Cortex-M平台编译部署](#armcortex-m平台编译部署)

<!-- /TOC -->

## 概述

本教程以MNIST分类模型推理代码为例，帮助用户了解codegen生成代码、编译构建、部署等流程。

## 模型编译体验

用户可以使用脚本一键式编译生成MNIST分类模型的推理代码并执行推理，得到单次推理输出。下载[MindSpore源码](https://gitee.com/mindspore/mindspore)，进入[`mindspore/mindspore/lite/micro/examples/mnist_x86`](https://gitee.com/mindspore/mindspore/tree/r1.5/mindspore/lite/micro/example/mnist_x86)目录，执行脚本`mnist.sh`自动生成模型推理代码并编译工程目录，即可得到单次推理输出。

```bash
bash mnist.sh
```

推理结果如下：

```text
======run benchmark======
input 0: mnist_input.bin

outputs:
name: Softmax-7, DataType: 43, Size: 40, Shape: [1 10], Data:
0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
======run success=======
```

也可以按照**详细步骤**从生成代码开始逐步完成使用codegen编译一个MNIST分类模型的全流程。

## 详细步骤

在编译此工程之前需要预先获取Ubuntu-x64 CPU平台的[Release包](https://www.mindspore.cn/lite/docs/zh-CN/r1.5/use/downloads.html)，解压后得到`mindspore-lite-{version}-linux-x64`，将其拷贝到当前目录下。

> `{version}`为版本号字符串，如`1.2.0`。

以本教程为例，预置x86平台的Release包目录如下：

```text
  mindspore-lite-{version}-linux-x64
 └── tools
     └── codegen # 代码生成工具
         ├── codegen                # 可执行程序
         ├── include                # 推理框架头文件
         │   ├── nnacl              # nnacl 算子头文件
         │   └── wrapper
         ├── lib
         │   └── libwrapper.a       # MindSpore Lite CodeGen生成代码依赖的部分算子静态库
         └── third_party
             ├── include
             │   └── CMSIS          # ARM CMSIS NN 算子头文件
             └── lib
                 └── libcmsis_nn.a  # ARM CMSIS NN 算子静态库
```

### 生成代码

下载[MNIST分类网络](https://download.mindspore.cn/model_zoo/official/lite/mnist_lite/mnist.ms)。使用Release包中的codegen编译MNIST分类模型，生成对应的x86平台推理代码。生成代码的具体命令如下：

```bash
./codegen --codePath=. --modelPath=mnist.ms --target=x86
```

codegen在当前目录下将生成mnist目录，其中包含了可编译构建的mnist分类模型的代码。
> 关于codegen的更多使用命令说明，可参见[codegen使用说明](https://www.mindspore.cn/lite/docs/zh-CN/r1.5/use/micro.html#id4)。

### 部署应用

接下来介绍如何构建MindSpore Lite CodeGen生成的模型推理代码工程，并在x86平台完成部署。上文中codegen生成的代码与`mindspore/mindspore/lite/micro/example/mnist_x86`相同，本章节编译、构建步骤将对该目录展开，用户也可参照相同操作，编译上文codegen生成mnist目录代码。

#### 编译依赖

- [CMake](https://cmake.org/download/) >= 3.18.3
- [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

#### 构建与运行

1. **生成代码工程说明**

   进入`mindspore/mindspore/lite/micro/example/mnist_x86`目录中。

   生成代码工程目录说明：

   当前目录下预置了MNIST分类网络生成的代码。

   ```text
   mnist_x86/                         # 生成代码的根目录
   ├── benchmark                      # 生成代码的benchmark目录
   └── src                            # 模型推理代码目录
   ```

2. **代码编译**

   组织模型生成的推理代码以及算子静态库，编译生成模型推理静态库并编译生成benchmark可执行文件,

   进入代码工程目录下，新建并进入build目录：

   ```bash
   mkdir build && cd build
   ```

   开始编译：

   ```bash
   cmake -DPKG_PATH={path to}/mindspore-lite-{version}-linux-x64 ..
   make
   ```

   > `{path to}`和`{version}`需要用户根据实际情况填写。

   代码工程编译成功结果：

   ```text
   Scanning dependencies of target net
   [ 12%] Building C object src/CMakeFiles/net.dir/net.c.o
   [ 25%] Building CXX object src/CMakeFiles/net.dir/session.cc.o
   [ 37%] Building CXX object src/CMakeFiles/net.dir/tensor.cc.o
   [ 50%] Building C object src/CMakeFiles/net.dir/weight.c.o
   [ 62%] Linking CXX static library libnet.a
   unzip raw static library libnet.a
   raw static library libnet.a size:
   -rw-r--r-- 1 user user 58K Mar 22 10:09 libnet.a
   generate specified static library libnet.a
   new static library libnet.a size:
   -rw-r--r-- 1 user user 162K Mar 22 10:09 libnet.a
   [ 62%] Built target net
   Scanning dependencies of target benchmark
   [ 75%] Building CXX object CMakeFiles/benchmark.dir/benchmark/benchmark.cc.o
   [ 87%] Building C object CMakeFiles/benchmark.dir/benchmark/load_input.c.o
   [100%] Linking CXX executable benchmark
   [100%] Built target benchmark
   ```

   此时在`mnist_x86/build/src/`目录下生成了`libnet.a`，推理执行库，在`mnist_x86/build`目录下生成了`benchmark`可执行文件。

3. **代码部署**

   本示例部署于x86平台。由代码工程编译成功以后的产物为`benchmark`可执行文件，将其拷贝到用户的目标Linux服务器中即可执行。

   在目标Linux服务上执行编译成功的二进制文件：

   ```bash
   ./benchmark mnist_input.bin net.bin
   ```

   > mnist_input.bin在`example/mnist_x86`目录下，`net.bin`为模型参数文件，在`example/mnist_x86/src`目录下。

   生成结果如下：

   ```text
   start run benchmark
   input 0: mnist_input.bin
   output size: 1
   uint8:
   Name: Softmax-7, DataType: 43, Size: 40, Shape: 1 10, Data:
   0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
   run benchmark success
   ```

#### 编写推理代码示例

本教程中的`benchmark`内部实现主要用于指导用户如何编写以及调用codegen编译的模型推理代码接口。以下为接口调用的详细介绍，详情代码可以参见[examples/mnist_x86](https://gitee.com/mindspore/mindspore/tree/r1.5/mindspore/lite/micro/example/mnist_x86)下的示例代码示例：

1. **构建推理的上下文以及会话**

   本教程生成的代码为非并行代码，无需上下文context，可直接设为空。

   ```cpp
     size_t model_size = 0;
     Context *context = nullptr;
     session::LiteSession *session = mindspore::session::LiteSession::CreateSession(model_buffer, model_size, context);
     if (session == nullptr) {
         std::cerr << "create lite session failed" << std::endl;
         return RET_ERROR;
     }
   ```

2. **输入数据准备**

   用户所需要准备的输入数据内存空间，若输入是持久化文件，可通过读文件方式获取。若输入数据已经存在内存中，则此处无需读取，可直接传入数据指针。

   ```cpp
     std::vector<MSTensor *> inputs = session->GetInputs();
     MSTensor *input = inputs.at(0);
     if (input == nullptr) {
         return RET_ERROR;
     }
     // Assume we have got input data in memory.
     memcpy(input->MutableData(), input_buffer, input->Size());
   ```

3. **执行推理**

   ```cpp
     session->RunGraph();
   ```

4. **推理结束获取输出**

   ```cpp
     Vector<String> outputs_name = session->GetOutputTensorNames();
     for (const auto &name : outputs_name) {
         auto output = session->GetOutputByTensorName(name);
         // deal with output
         ......
     }
   ```

5. **释放内存session**

   ```cpp
     delete session;
   ```

6. **推理代码整体调用流程**

   ```cpp
     // Assume we have got model_buffer data in memory.
     size_t model_size = 0;
     Context *context = nullptr;
     session::LiteSession *session = mindspore::session::LiteSession::CreateSession(model_buffer, model_size, context);
     if (session == nullptr) {
         std::cerr << "create lite session failed" << std::endl;
         return RET_ERROR;
     }

     std::vector<MSTensor *> inputs = session->GetInputs();
     MSTensor *input = inputs.at(0);
     if (input == nullptr) {
         return RET_ERROR;
     }
     // Assume we have got input data in memory.
     memcpy(input->MutableData(), input_buffer, input->Size());

     session->RunGraph();

     Vector<String> outputs_name = session->GetOutputTensorNames();
     for (const auto &name : outputs_name) {
         auto output = session->GetOutputByTensorName(name);
         // deal with output
         ......
     }

     delete session;
   ```

## 更多详情

### [Android平台编译部署](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/lite/micro/example/mobilenetv2/README.md#)

### [Arm&nbsp;Cortex-M平台编译部署](https://www.mindspore.cn/lite/docs/zh-CN/r1.5/use/micro.html)
