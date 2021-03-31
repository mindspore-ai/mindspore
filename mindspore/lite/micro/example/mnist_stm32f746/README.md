

# Arm Cortex-M编译部署

 `Linux` `Cortex-M`  `IOT` `C/C++` `全流程` `模型编译` `模型代码生成` `模型部署` `推理应用` `初级` `中级` `高级`

<!-- TOC -->

- Arm Cortex-M编译部署
    - [STM32F746编译依赖](#STM32F746编译依赖)
    - [STM32F746构建](#STM32F746构建)
    - [STM32F746工程部署](#STM32F746工程部署)
    - [更多详情](#更多详情)
        - [Linux_x86_64编译部署](#Linux_x86_64编译部署)
        - [Android平台编译部署](#Android平台编译部署)

<!-- /TOC -->

## Arm Cortex-M编译部署

本教程以在STM32F746单板上编译部署生成模型代码为例，演示了codegen编译模型在Cortex-M平台的使用。更多关于Arm Cortex-M的详情可参见其[官网](https://developer.arm.com/ip-products/processors/cortex-m)。

### STM32F746编译依赖

模型推理代码的编译部署需要在windows上安装[Jlink]((https://www.segger.com/))、[STM32CubeMX](https://www.st.com/content/st_com/en.html)、[gcc-arm-none-ebai](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm)等工具来进行交叉编译。

- [STM32CubeMX-Win](https://www.st.com/content/ccc/resource/technical/software/sw_development_suite/group0/0b/05/f0/25/c7/2b/42/9d/stm32cubemx_v6-1-1/files/stm32cubemx_v6-1-1.zip/jcr:content/translations/en.stm32cubemx_v6-1-1.zip) >= 6.0.1

- [gcc-arm-none-eabi](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads)  >= 9-2019-q4-major-win32

- [JLink-windows](https://www.segger.com/downloads/jlink/) >= 6.56
- [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
- [CMake](https://cmake.org/download/) >= 3.18.3

### STM32F746构建

首先使用codegen编译MNIST手写数字识别模型，生成对应的STM32F46推理代码。具体命令如下：

```bash
./codegen --codePath=. --modelPath=mnist.ms --target=ARM32M
```

#### 代码工程说明

```bash
├── MNIST
└── include            # 模型推理对外API头文件目录
└── operator_library

```

##### 算子相关目录说明

在编译此工程之前需要预先获取对应平台所需要的算子文件，由于Cortex-M平台工程编译一般涉及到较复杂的交叉编译，此处不提供直接预编译的算子库静态库，而是用户根据模型自行组织文件，自主编译Cortex-M7 、Coretex-M4、Cortex-M3等工程(对应工程目录结构已在示例代码中给出，用户可自主将对应arm官方的CMSIS源码放置其中即可)。

预置算子静态库的目录如下:

```bash
├── operator_library    # 对应平台算子库目录
    ├── include         # 对应平台算子库头文件目录
    └── nnacl           # 对应mindspore团队提供的平台算子库源文件
    └── wrapper         # 对应mindspore团队提供的平台算子库源文件
    └── CMSIS           # 对应Arm官方提供的CMSIS平台算子库源文件
    
```

  > 在使用过程中，我们注意到引入Softmax相关的CMSIS算子文件时，头文件中需要加入`arm_nnfunctions.h`,使用者可以稍作注意。

生成代码工程目录如下：

模型推理对外API头文件可由mindspore团队发布的[Release包](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/downloads.html)中获取。

```bash
├── MNIST               # 生成代码的根目录
    ├── benchmark       # 生成代码的benchmark目录
    └── src             # 模型推理代码目录
```

#### 代码工程编译

##### 环境测试

安装好交叉编译所需环境后，需要在windows环境中依次将其加入到环境变量中

```bash
gcc -v               # 查看GCC版本
arm-none-eabi-gdb -v # 查看交叉编译环境
jlink -v             # 查看jlink版本
make -v              # 查看make版本
```

以上的命令均有成功返回值时，表明环境准备ok，可以继续进入下一步，否则先安装上述环境！！！

##### 生成STM32F746单板初始化代码([详情示例代码](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/micro/example/mnist_stm32f746))

1. 启动 STM32CubeMX，新建project，选择单板STM32F746IG

2. 成功以后，选择`Makefile` ，`generator code`

3. 在生成的工程目录下打开`cmd`，执行`make`，测试初始代码是否成功编译。

   ```bash
   # make成功结果
   arm-none-eabi-size build/test_stm32f746.elf
      text    data     bss     dec     hex filename
      3660      20    1572    5252    1484 build/test_stm32f746.elf
   arm-none-eabi-objcopy -O ihex build/test_stm32f746.elf build/test_stm32f746.hex
   arm-none-eabi-objcopy -O binary -S build/test_stm32f746.elf build/test_stm32f746.bin
   ```

##### 编译模型

1. 拷贝mindspore团队提供算子文件以及对应头文件到STM32CubeMX生成的工程目录中。

2. 拷贝codegen生成模型推理代码到 STM32CubeMX生成的代码工程目录中

   ```bash
   ├── .mxproject
   └── build             # 工程编译输出目录
   └── Core
   └── Drivers
   └── mnist             # codegen生成的cortex-m7 模型推理代码
   └── Makefile          # 组织工程makefile文件需要用户自己修改组织mnist && operator_library到工程目录中
   └── startup_stm32f746xx.s
   └── STM32F746IGKx_FLASH.ld
   └── test_stm32f746.ioc
   ```
   
3. 修改makefile文件，组织算子静态库以及模型推理代码,具体makefile文件内容参见[示例](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/micro/example/mnist_stm32f746)。

   ```bash
   # C includes
   C_INCLUDES =  \
   -ICore/Inc \
   -IDrivers/STM32F7xx_HAL_Driver/Inc \
   -IDrivers/STM32F7xx_HAL_Driver/Inc/Legacy \
   -IDrivers/CMSIS/Device/ST/STM32F7xx/Include \
   -Imnist/operator_library/include \                # 新增，指定算子库头文件目录
   -Imnist/include \                           	  # 新增，指定模型推理代码头文件
   -Imnist/src                                 	  # 新增，指定模型推理代码头文件
   ```
   
4. 在工程目录的Core/Src的main.c编写模型调用代码，具体代码新增如下：

   ```cpp
   while (1) {
       /* USER CODE END WHILE */
       SEGGER_RTT_printf(0, "***********mnist test start***********\n");
       const char *model_buffer = nullptr;
       int model_size = 0;
       session::LiteSession *session = mindspore::session::LiteSession::CreateSession(model_buffer, 		     model_size, nullptr);
       Vector<tensor::MSTensor *> inputs = session->GetInputs();
       size_t inputs_num = inputs.size();
       void *inputs_binbuf[inputs_num];
       int inputs_size[inputs_num];
       for (size_t i = 0; i < inputs_num; ++i) {
         inputs_size[i] = inputs[i]->Size();
       }
       // here mnist only have one input data,just hard code to it's array;
       inputs_binbuf[0] = mnist_inputs_data;
       for (size_t i = 0; i < inputs_num; ++i) {
         void *input_data = inputs[i]->MutableData();
         memcpy(input_data, inputs_binbuf[i], inputs_size[i]);
       }
       int ret = session->RunGraph();
       if (ret != lite::RET_OK) {
         return lite::RET_ERROR;
       }
       Vector<String> outputs_name = session->GetOutputTensorNames();
       for (int i = 0; i < outputs_name.size(); ++i) {
         tensor::MSTensor *output_tensor = session->GetOutputByTensorName(outputs_name[i]);
         if (output_tensor == nullptr) {
           return -1;
         }
         float *casted_data = static_cast<float *>(output_tensor->MutableData());
         if (casted_data == nullptr) {
           return -1;
         }
         for (size_t j = 0; j < 10 && j < output_tensor->ElementsNum(); j++) {
           SEGGER_RTT_printf(0, "output: [%d] is : [%d]/100\n", i, casted_data[i] * 100);
         }
       }
       delete session;
       SEGGER_RTT_printf(0, "***********mnist test end***********\n");
   ```

5. 在工程跟目中目录使用管理员权限打开`cmd` 执行 `make`进行编译

   ```bash
   make
   ```

### STM32F746工程部署

使用jlink 将可执行文件拷贝到单板上并做推理

```bash
jlinkgdbserver           # 启动jlinkgdbserver 选定target device为STM32F746IG
jlinkRTTViewer           # 启动jlinkRTTViewer 选定target devices为STM32F746IG
arm-none-eabi-gdb        # 启动arm-gcc gdb服务
file build/target.elf    # 打开调测文件
target remote 127.0.0.1  # 连接jlink服务器
monitor reset            # 重置单板
monitor halt             # 挂起单板
load                     # 加载可执行文件到单板
c                        # 执行模型推理
```

## 更多详情

### [Linux_x86_64平台编译部署](https://www.mindspore.cn/tutorial/lite/zh-CN/master/quick_start/quick_start_codegen.html)

### [Android平台编译部署](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/micro/example/mobilenetv2)

