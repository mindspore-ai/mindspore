# Android编译部署

 `Linux` `Cortex-M`  `IOT` `C/C++` `全流程` `模型编译` `模型代码生成` `模型部署` `推理应用` `初级` `中级` `高级`

<!-- TOC -->

- Arm Cortex-M编译部署
    - [STM32F746编译依赖](#STM32F746编译依赖)
    - [STM32F746构建](#STM32F746构建)
    - [STM32F746工程部署](#STM32F746工程部署)
    - [更多详情](#更多详情)
        - [Linux x86_64平台编译部署](#Linux x86_64平台编译部署)
        - [Android平台编译部署](#STM32746平台编译部署)

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

### STM32F746构建与运行

首先使用codegen编译LeNet模型，生成对应的STM32F46推理代码。具体命令如下：

```bash
./codegen --codePath=. --modelPath=LeNet.ms --moduleName=LeNet --target=ARM32M
```

#### 代码工程说明

```bash
├── LeNet
└── operator_library
```

##### 算子静态库目录说明

在编译此工程之前需要预先获取Cortex-M 平台对应的[算子库]()。

预置算子静态库的目录如下:

```bash
├── operator_library    # 对应平台算子库目录
    ├── include         # 对应平台算子库头文件目录
    └── lib             # 对应平台算子库静态库目录
```

生成代码工程目录如下：

```bash
├── LeNet               # 生成代码的根目录
    ├── benchmark       # 生成代码的benchmark目录
    ├── include         # 模型推理代码对外暴露头文件目录
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

##### 生成STM32F746单板初始化代码([详情示例代码]())

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

##### 编译生成模型静态库

1. 拷贝mindspore团队提供的cortex-m7的算子静态库以及对应头文件到STM32CubeMX生成的工程目录中。

2. 拷贝codegen生成模型推理代码到 STM32CubeMX生成的代码工程目录中

   ```bash
   ├── .mxproject
   └── build             # 工程编译目录最终的elf文件存在于此
   └── Core
   └── Drivers
   └── LeNet             # codegen生成的cortex-m7 模型推理代码
   └── Makefile          # 组织工程makefile文件需要用户自己修改组织lenet && operator_library到工程目录中
   └── operator_library  # mindspore团队提供的对应平台算子库
   └── startup_stm32f746xx.s
   └── STM32F746IGKx_FLASH.ld
   └── test_stm32f746.ioc
   ```

3. 修改makefile文件，组织算子静态库以及模型推理代码

   ```bash
   # C includes
   C_INCLUDES =  \
   -ICore/Inc \
   -IDrivers/STM32F7xx_HAL_Driver/Inc \
   -IDrivers/STM32F7xx_HAL_Driver/Inc/Legacy \
   -IDrivers/CMSIS/Device/ST/STM32F7xx/Include \
   -Ioperator_library/include \                # 新增，指定算子库头文件目录
   -ILeNet/include \                           # 新增，指定模型推理代码头文件
   -ILeNet/src                                 # 新增，指定模型推理代码头文件
   # libraries
   LIBS = -lc -lm -lnosys -lops                # 修改，导入mindspore团队提供算子库
   LIBDIR = -Ioperator_library/lib/arm32m      # 新增，指定算子库所在路径
   ```

4. 在工程目录的Core/Src的main.c编写模型调用代码，具体代码新增如下：

   ```cpp
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

#### 执行结果

```bash
```

## 更多详情

### [Linux x86_64平台编译部署]()

### [Android平台编译部署]()

