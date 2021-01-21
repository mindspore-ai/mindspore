# MindSpore Lite 端侧图像分割demo（Android）

本示例程序演示了如何在端侧利用MindSpore Lite Java API 以及MindSpore Lite 图像分割模型完成端侧推理，实现对设备摄像头捕获的内容进行分割，并在App图像预览界面中显示出最可能的分割结果。

## 运行依赖

- Android Studio >= 3.2 (推荐4.0以上版本)

## 构建与运行

1. 在Android Studio中加载本示例源码。

    ![start_home](images/home.png)

    启动Android Studio后，点击`File->Settings->System Settings->Android SDK`，勾选相应的`SDK Tools`。如下图所示，勾选后，点击`OK`，Android Studio即可自动安装SDK。

    ![start_sdk](images/sdk_management.jpg)

    > Android SDK Tools为默认安装项，取消`Hide Obsolete Packages`选框之后可看到。
    >
    > 使用过程中若出现问题，可参考第4项解决。

2. 连接Android设备，运行该应用程序。

    通过USB连接Android手机。待成功识别到设备后，点击`Run 'app'`即可在您的手机上运行本示例项目。

    > 编译过程中Android Studio会自动下载MindSpore Lite、模型文件等相关依赖项，编译过程需做耐心等待。
    >
    > Android Studio连接设备调试操作，可参考<https://developer.android.com/studio/run/device?hl=zh-cn>。
    >
    > 手机需开启“USB调试模式”，Android Studio 才能识别到手机。 华为手机一般在设置->系统和更新->开发人员选项->USB调试中开始“USB调试模型”。

    ![run_app](images/run_app.PNG)

3. 在Android设备上，点击“继续安装”，安装完即可查看到设备摄像头捕获的内容和推理结果。

    ![install](images/install.jpg)

    如下图所示，识别出的概率最高的物体是植物。

    ![result](images/app_result.jpg)

4. Demo部署问题解决方案。

    4.1 NDK、CMake、JDK等工具问题：

    如果Android Studio内安装的工具出现无法识别等问题，可重新从相应官网下载和安装，并配置路径。

    - NDK >= 21.3 [NDK](https://developer.android.google.cn/ndk/downloads?hl=zh-cn)
    - CMake >= 3.10.2   [CMake](https://cmake.org/download)
    - Android SDK >= 26 [SDK](https://developer.microsoft.com/zh-cn/windows/downloads/windows-10-sdk/)
    - JDK >= 1.8 [JDK](https://www.oracle.com/cn/java/technologies/javase/javase-jdk8-downloads.html)

        ![project_structure](images/project_structure.png)

    4.2 NDK版本不匹配问题：

    打开`Android SDK`，点击`Show Package Details`，根据报错信息选择安装合适的NDK版本。
    ![NDK_version](images/NDK_version.jpg)

    4.3 Android Studio版本问题：

    在`工具栏-help-Checkout for Updates`中更新Android Studio版本。

    4.4 Gradle下依赖项安装过慢问题：

   如图所示， 打开Demo根目录下`build.gradle`文件，加入华为镜像源地址：`maven {url 'https://developer.huawei.com/repo/'}`，修改classpath为4.0.0，点击`sync`进行同步。下载完成后，将classpath版本复原，再次进行同步。
    ![maven](images/maven.jpg)

## 示例程序详细说明  

本端侧图像分割Android示例程序使用Java实现，Java层主要通过Android Camera 2 API实现摄像头获取图像帧，进行相应的图像处理，之后调用Java API 完成模型推理。

> 此处详细说明示例程序的Java层图像处理及模型推理实现，Java层运用Android Camera 2 API实现开启设备摄像头以及图像帧处理等功能，需读者具备一定的Android开发基础知识。

### 示例程序结构

```text
app
├── src/main
│   ├── assets # 资源文件
|   |   └── deeplabv3.ms # 存放模型文件
│   |
│   ├── java # java层应用代码
│   │   └── com.mindspore.imagesegmentation
│   │       ├── help # 图像处理及MindSpore Java调用相关实现
│   │       │   └── ImageUtils # 图像预处理
│   │       │   └── ModelTrackingResult # 推理数据后处理
│   │       │   └── TrackingMobile # 模型加载、构建计算图和推理
│   │       └── BitmapUtils # 图像处理
│   │       └── MainActivity # 交互主页面
│   │       └── OnBackgroundImageListener # 获取相册图像
│   │       └── StyleRecycleViewAdapter # 获取相册图像
│   │
│   ├── res # 存放Android相关的资源文件
│   └── AndroidManifest.xml # Android配置文件
│
├── CMakeList.txt # cmake编译入口文件
│
├── build.gradle # 其他Android配置文件
├── download.gradle # 工程依赖文件下载
└── ...
```

### 配置MindSpore Lite依赖项

Android 调用MindSpore Java API时，需要相关库文件支持。可通过MindSpore Lite[源码编译](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html)生成`mindspore-lite-{version}-minddata-{os}-{device}.tar.gz`库文件包并解压缩（包含`libmindspore-lite.so`库文件和相关头文件），在本例中需使用生成带图像预处理模块的编译命令。

> version：输出件版本号，与所编译的分支代码对应的版本一致。
>
> device：当前分为cpu（内置CPU算子）和gpu（内置CPU和GPU算子）。
>
> os：输出件应部署的操作系统。

本示例中，build过程由download.gradle文件自动下载MindSpore Lite 版本文件，并放置在`app/src/main/cpp/`目录下。

> 若自动下载失败，请手动下载相关库文件，解压并放在对应位置：

  mindspore-lite-1.0.1-runtime-arm64-cpu.tar.gz [下载链接](https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.0.1/lite/android_aarch64/mindspore-lite-1.0.1-runtime-arm64-cpu.tar.gz)

在app的`build.gradle`文件中配置CMake编译支持，以及`arm64-v8a`的编译支持，如下所示：

```text
android{
    defaultConfig{
        externalNativeBuild{
            cmake{
                arguments "-DANDROID_STL=c++_shared"
            }
        }

        ndk{
            abiFilters 'arm64-v8a'
        }
    }
}
```

在`app/CMakeLists.txt`文件中建立`.so`库文件链接，如下所示。

```text
# ============== Set MindSpore Dependencies. =============
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/third_party/flatbuffers/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION})
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/include/ir/dtype)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/include/schema)

add_library(mindspore-lite SHARED IMPORTED )
add_library(minddata-lite SHARED IMPORTED )

set_target_properties(mindspore-lite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/lib/libmindspore-lite.so)
set_target_properties(minddata-lite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/lib/libminddata-lite.so)
# --------------- MindSpore Lite set End. --------------------

# Link target library.
target_link_libraries(
    ...
     # --- mindspore ---
        minddata-lite
        mindspore-lite
    ...
)
```

### 下载及部署模型文件

从MindSpore Model Hub中下载模型文件，本示例程序中使用的终端图像分割模型文件为`deeplabv3.ms`，同样通过download.gradle脚本在APP构建时自动下载，并放置在`app/src/main/assets`工程目录下。

> 若下载失败请手动下载模型文件，deeplabv3.ms [下载链接](https://download.mindspore.cn/model_zoo/official/lite/deeplabv3_lite/deeplabv3.ms)。

### 编写端侧推理代码

调用MindSpore Lite Java API实现端测推理。

推理代码流程如下，完整代码请参见`src/java/TrackingMobile.java`。

1. 加载MindSpore Lite模型文件，构建上下文、会话以及用于推理的计算图。

    - 加载模型文件：创建并配置用于模型推理的上下文

    ```Java
    // Create context and load the .ms model named 'IMAGESEGMENTATIONMODEL'
    model = new Model();
    if (!model.loadModel(Context, IMAGESEGMENTATIONMODEL)) {
      Log.e(TAG, "Load Model failed");
      return;
    }
    ```

    - 创建会话

    ```Java
    // Create and init config.
    msConfig = new MSConfig();
    if (!msConfig.init(DeviceType.DT_CPU, 2, CpuBindMode.MID_CPU)) {
      Log.e(TAG, "Init context failed");
      return;
    }

    // Create the MindSpore lite session.
    session = new LiteSession();
    if (!session.init(msConfig)) {
      Log.e(TAG, "Create session failed");
      msConfig.free();
      return;
    }
    msConfig.free();
    ```

    - 构建计算图

    ```Java
    if (!session.compileGraph(model)) {
      Log.e(TAG, "Compile graph failed");
      model.freeBuffer();
      return;
    }
    // Note: when use model.freeBuffer(), the model can not be compile graph again.
    model.freeBuffer();
    ```

2. 将输入图片转换为传入MindSpore模型的Tensor格式。

    ```Java
    List<MSTensor> inputs = session.getInputs();
    if (inputs.size() != 1) {
      Log.e(TAG, "inputs.size() != 1");
      return null;
    }

    // `bitmap` is the picture used to infer.
    float resource_height = bitmap.getHeight();
    float resource_weight = bitmap.getWidth();
    ByteBuffer contentArray = bitmapToByteBuffer(bitmap, imageSize, imageSize, IMAGE_MEAN, IMAGE_STD);

    MSTensor inTensor = inputs.get(0);
    inTensor.setData(contentArray);
    ```

3. 对输入Tensor按照模型进行推理，获取输出Tensor，并进行后处理。

    - 图执行，端侧推理。

    ```Java
    // After the model and image tensor data is loaded, run inference.
    if (!session.runGraph()) {
      Log.e(TAG, "Run graph failed");
      return null;
    }
    ```

    - 获取输出数据。

    ```Java
    // Get output tensor values, the model only outputs one tensor.
    List<String> tensorNames = session.getOutputTensorNames();
    MSTensor output = session.getOutputByTensorName(tensorNames.front());
    if (output == null) {
      Log.e(TAG, "Can not find output " + tensorName);
      return null;
    }
    ```

    - 输出数据的后续处理。

    ```Java
    // Show output as pictures.
    float[] results = output.getFloatData();

    ByteBuffer bytebuffer_results = floatArrayToByteArray(results);

    Bitmap dstBitmap = convertBytebufferMaskToBitmap(bytebuffer_results, imageSize, imageSize, bitmap, dstBitmap, segmentColors);
    dstBitmap = scaleBitmapAndKeepRatio(dstBitmap, (int) resource_height, (int) resource_weight);
    ```

4. 图片处理及输出数据后处理请参考如下代码。

    ```Java
    Bitmap scaleBitmapAndKeepRatio(Bitmap targetBmp, int reqHeightInPixels, int reqWidthInPixels) {
      if (targetBmp.getHeight() == reqHeightInPixels && targetBmp.getWidth() == reqWidthInPixels) {
        return targetBmp;
      }

      Matrix matrix = new Matrix();
      matrix.setRectToRect(new RectF(0f, 0f, targetBmp.getWidth(), targetBmp.getHeight()),
                           new RectF(0f, 0f, reqWidthInPixels, reqHeightInPixels), Matrix.ScaleToFit.FILL;

        return Bitmap.createBitmap(targetBmp, 0, 0, targetBmp.getWidth(), targetBmp.getHeight(), matrix, true);
    }

    ByteBuffer bitmapToByteBuffer(Bitmap bitmapIn, int width, int height, float mean, float std) {
      Bitmap bitmap = scaleBitmapAndKeepRatio(bitmapIn, width, height);
      ByteBuffer inputImage = ByteBuffer.allocateDirect(1 * width * height * 3 * 4);
      inputImage.order(ByteOrder.nativeOrder());
      inputImage.rewind();
      int[] intValues = new int[width * height];
      bitmap.getPixels(intValues, 0, width, 0, 0, width, height);
      int pixel = 0;
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          int value = intValues[pixel++];
          inputImage.putFloat(((float) (value >> 16 & 255) - mean) / std);
          inputImage.putFloat(((float) (value >> 8 & 255) - mean) / std);
          inputImage.putFloat(((float) (value & 255) - mean) / std);
        }
      }
      inputImage.rewind();
      return inputImage;
    }

    ByteBuffer floatArrayToByteArray(float[] floats) {
      ByteBuffer buffer = ByteBuffer.allocate(4 * floats.length);
      FloatBuffer floatBuffer = buffer.asFloatBuffer();
      floatBuffer.put(floats);
      return buffer;
    }

    Bitmap convertBytebufferMaskToBitmap(ByteBuffer inputBuffer, int imageWidth, int imageHeight, Bitmap backgroundImage, int[] colors) {
      Bitmap.Config conf = Bitmap.Config.ARGB_8888;
      Bitmap dstBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf);
      Bitmap scaledBackgroundImage = scaleBitmapAndKeepRatio(backgroundImage, imageWidth, imageHeight);
      int[][] mSegmentBits = new int[imageWidth][imageHeight];
      inputBuffer.rewind();
      for (int y = 0; y < imageHeight; y++) {
        for (int x = 0; x < imageWidth; x++) {
          float maxVal = 0f;
          mSegmentBits[x][y] = 0;
          // NUM_CLASSES is the number of labels, the value here is 21.
          for (int i = 0; i < NUM_CLASSES; i++) {
            float value = inputBuffer.getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + i) * 4);
            if (i == 0 || value > maxVal) {
              maxVal = value;
              // Check whether a pixel belongs to a person whose label is 15.
              if (i == 15) {
                mSegmentBits[x][y] = i;
              } else {
                mSegmentBits[x][y] = 0;
              }
            }
          }
          itemsFound.add(mSegmentBits[x][y]);

          int newPixelColor = ColorUtils.compositeColors(
                  colors[mSegmentBits[x][y] == 0 ? 0 : 1],
                  scaledBackgroundImage.getPixel(x, y)
          );
          dstBitmap.setPixel(x, y, mSegmentBits[x][y] == 0 ? colors[0] : scaledBackgroundImage.getPixel(x, y));
        }
      }
      return dstBitmap;
    }
    ```
