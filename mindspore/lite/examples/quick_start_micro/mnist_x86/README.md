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

本教程以MNIST分类模型为例，介绍面向边缘侧设备超轻量AI推理引擎Micro，包括模型自动生成代码、编译构建、部署等三步。

## 模型编译体验

用户可以使用脚本一键式编译生成MNIST分类模型的推理代码并执行推理，得到单次推理输出。

第一步：下载MindSpore源码(https://gitee.com/mindspore/mindspore)，在项目根目录使用命令

```bash
bash build.sh -I x86_64 -j128
```

编译MindSpore，在项目根目录的output目录会生成MindSpore tar 包。

第二步：进入mindspore/mindspore/lite/examples/quick_start_micro/mnist_x86 目录，执行脚本

```bash
bash mnist.sh -g on -r ${dir}/mindspore-lite-${VERSION_STR}-linux-x64.tar.gz
```

自动生成模型推理代码并编译工程目录，即可得到单次推理输出。推理结果如下：

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

**模型编译体验**第一步中编译的x86平台tar包目录如下：

```text
  mindspore-lite-{version}-linux-x64
 └── tools
     └── codegen # 代码生成工具
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

**模型编译体验**第二步中会先下载[MNIST分类网络](https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/mnist.tar.gz)，  模型及输入输出标杆数据解压在 quick_start_micro/models 目录下。
使用 Release 包中的 codegen 编译 MNIST 分类模型，生成对应的 x86 平台推理代码，具体命令如下：

```shell
./converter_lite --fmk=TFLITE --modelFile=${model_dir}/mnist.tflite --outputFile=${SOURCE_CODE_DIR} --configFile=${COFIG_FILE}
```

outputFile 指定micro代码生成目录，当前脚本目录下将生成source_code目录，其中包含了可编译构建的mnist分类模型的代码src和benchmark用例。
> 关于codegen的更多使用命令说明，可参见[codegen使用说明](https://www.mindspore.cn/lite/docs/zh-CN/r1.9/use/micro.html#自动生成的代码部署时依赖的头文件和lib的目录结构)。

### 部署应用

接下来介绍如何构建MindSpore Lite Micro生成的模型推理代码工程，并在x86平台完成部署。

#### 编译依赖

- [CMake](https://cmake.org/download/) >= 3.18.3
- [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

#### 构建与运行

1. **生成代码工程说明**

   进入`mindspore/mindspore/lite/example/quick_start_micro/mnist_x86`目录中,生成代码工程目录说明：
   
   ```text
   source_code/                       # 生成代码的根目录
   ├── benchmark                      # 生成代码的benchmark目录
   └── src                            # 模型推理代码目录
   ```

2. **代码编译**

   组织模型生成的推理代码以及算子静态库，编译生成模型推理静态库并编译生成benchmark可执行文件, 进入代码工程目录下，新建并进入build目录：
   
   ```bash
   cd source_code
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
   
   此时在`mnist_x86/source_code/build/src/`目录下生成了`libnet.a`，推理执行库，在`mnist_x86/build`目录下生成了`benchmark`可执行文件。

3. **代码部署**

   本示例部署于x86平台。由代码工程编译成功以后的产物为`benchmark`可执行文件，将其拷贝到用户的目标Linux服务器中即可执行。

   在目标Linux服务上执行编译成功的二进制文件：
   
   ```bash
   ./benchmark mnist.tflite.ms.bin net.bin mnist.tflite.ms.out
   ```

   > mnist.tflite.ms.bin在`example/mnist_x86`目录下，`net.bin`为模型参数文件，在`example/mnist_x86/src`目录下。

   生成结果如下：

   ```text
   start run benchmark
   input 0: mnist.tflite.ms.bin
   output size: 1
   uint8:
   Name: Softmax-7, DataType: 43, Size: 40, Shape: 1 10, Data:
   0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
   run benchmark success
   ```

#### 编写推理代码示例

本教程中的`benchmark`内部实现主要用于指导用户如何编写以及调用codegen编译的模型推理代码接口。以下为接口调用的详细介绍，
详情代码可以参见[examples/quick_start_micro/mnist_x86](https://gitee.com/mindspore/mindspore/tree/r1.9/mindspore/lite/examples/quick_start_micro/mnist_x86)下的示例代码示例：

1. **构建推理的上下文以及会话**

   本教程生成的代码为非并行代码，无需上下文context，可直接设为空。

   ```cpp
     MSContextHandle ms_context_handle = NULL;
     void int model_size = 0;
     // read .bin file by ReadBinaryFile;
     model_buffer = ReadInputData("net.bin", &model_size);
     MSModelHandle model_handle = MSModelCreate();
     int ret = MSModelBuild(model_handle, model_buffer, model_size, kMSModelTypeMindIR, ms_context_handle);
     MSContextDestroy(&ms_context_handle);
   ```

2. **输入数据准备**

   用户所需要准备的输入数据内存空间，若输入是持久化文件，可通过读文件方式获取。若输入数据已经存在内存中，则此处无需读取，可直接传入数据指针。

   ```cpp
     for (size_t i = 0; i < inputs_num; ++i) {
       void *input_data = MSTensorGetMutableData(inputs_handle.handle_list[i]);
       memcpy(input_data, inputs_binbuf[i], inputs_size[i]);
       free(inputs_binbuf[i]);
       inputs_binbuf[i] = NULL;
     }
   ```

3. **执行推理**

   ```cpp
     MSTensorHandleArray outputs_handle = MSModelGetOutputs(model_handle);
     MSModelPredict(model_handle, inputs_handle, &outputs_handle, NULL, NULL);
   ```

4. **推理结束获取输出**

   ```cpp
     for (size_t i = 0; i < outputs_handle.handle_num; i++) {
       MSTensorHandle output = outputs_handle.handle_list[i];
       PrintTensorHandle(output);
     }
   ```

5. **释放内存session**

   ```cpp
     MSModelDestroy(&model_handle);
   ```

6. **推理代码整体调用流程**

   ```cpp
     // Assume we have got model_buffer data in memory.
     MSContextHandle ms_context_handle = NULL;
     void int model_size = 0;
     // read .bin file by ReadBinaryFile;
     model_buffer = ReadInputData("net.bin", &model_size);
     MSModelHandle model_handle = MSModelCreate();
     int ret = MSModelBuild(model_handle, model_buffer, model_size, kMSModelTypeMindIR, ms_context_handle);
     MSContextDestroy(&ms_context_handle);

     for (size_t i = 0; i < inputs_num; ++i) {
       void *input_data = MSTensorGetMutableData(inputs_handle.handle_list[i]);
       memcpy(input_data, inputs_binbuf[i], inputs_size[i]);
       free(inputs_binbuf[i]);
       inputs_binbuf[i] = NULL;
     }

     MSTensorHandleArray outputs_handle = MSModelGetOutputs(model_handle);
     MSModelPredict(model_handle, inputs_handle, &outputs_handle, NULL, NULL);

     for (size_t i = 0; i < outputs_handle.handle_num; i++) {
       MSTensorHandle output = outputs_handle.handle_list[i];
       PrintTensorHandle(output);
     }

     MSModelDestroy(&model_handle);
   ```

## 更多详情

### [Android平台编译部署](https://gitee.com/mindspore/mindspore/blob/r1.9/mindspore/lite/examples/quick_start_micro/mobilenetv2_arm64/README.md)

### [Arm&nbsp;Cortex-M平台编译部署](https://www.mindspore.cn/lite/docs/zh-CN/r1.9/use/micro.html)
