# MindSpore Lite 端侧风格迁移demo（Android）

本示例程序演示了如何在端侧利用MindSpore Lite API以及MindSpore Lite风格迁移模型完成端侧推理，根据demo内置的标准图片更换目标图片的艺术风格，并在App图像预览界面中显示出来。

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

风格Android示例程序通过Android Camera 2 API实现摄像头获取图像帧，以及相应的图像处理等功能，在[Runtime](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/runtime.html)中完成模型推理的过程。

### 示例程序结构

```text

├── app
│   ├── build.gradle # 其他Android配置文件
│   ├── download.gradle # APP构建时由gradle自动从HuaWei Server下载依赖的库文件及模型文件
│   ├── proguard-rules.pro
│   └── src
│       ├── main
│       │   ├── AndroidManifest.xml # Android配置文件
│       │   ├── java # java层应用代码
│       │   │   └── com
│       │   │       └── mindspore
│       │   │           └── posenetdemo # 图像处理及推理流程实现
│       │   │               ├── CameraDataDealListener.java
│       │   │               ├── ImageUtils.java
│       │   │               ├── MainActivity.java
│       │   │               ├── PoseNetFragment.java
│       │   │               ├── Posenet.java #
│       │   │               └── TestActivity.java
│       │   └── res # 存放Android相关的资源文件
│       └── test
└── ...

```

### 下载及部署模型文件

从MindSpore Model Hub中下载模型文件，本示例程序中使用的目标检测模型文件为`style_predict_quant.ms`、`style_transfer_quant.ms`，同样通过`download.gradle`脚本在APP构建时自动下载，并放置在`app/src/main/assets`工程目录下。

> 若下载失败请手动下载模型文件，style_predict_quant.ms [下载链接](https://download.mindspore.cn/model_zoo/official/lite/style_lite/style_predict_quant.ms)，以及style_transfer_quant.ms [下载链接](https://download.mindspore.cn/model_zoo/official/lite/style_lite/style_transfer_quant.ms)。

### 编写端侧推理代码

在风格迁移demo中，使用Java API实现端测推理。相比于C++ API，Java API可以直接在Java Class中调用，无需实现JNI层的相关代码，具有更好的便捷性。

风格迁移demo推理代码流程如下，完整代码请参见：`src/main/java/com/mindspore/styletransferdemo/StyleTransferModelExecutor.java`。

1. 加载MindSpore Lite模型文件，构建上下文、会话以及用于推理的计算图。

    - 加载模型：从文件系统中读取MindSpore Lite模型，并进行模型解析。

        ```java
        // Load the .ms model.
        style_predict_model = new Model();
        if (!style_predict_model.loadModel(mContext, "style_predict_quant.ms")) {
            Log.e("MS_LITE", "Load style_predict_model failed");
        }

        style_transform_model = new Model();
        if (!style_transform_model.loadModel(mContext, "style_transfer_quant.ms")) {
            Log.e("MS_LITE", "Load style_transform_model failed");
        }
        ```

    - 创建配置上下文：创建配置上下文`MSConfig`，保存会话所需的一些基本配置参数，用于指导图编译和图执行。

        ```java
        msConfig = new MSConfig();
        if (!msConfig.init(DeviceType.DT_CPU, NUM_THREADS, CpuBindMode.MID_CPU)) {
            Log.e("MS_LITE", "Init context failed");
        }
        ```

    - 创建会话：创建`LiteSession`，并调用`init`方法将上一步得到`MSConfig`配置到会话中。

        ```java
        // Create the MindSpore lite session.
        Predict_session = new LiteSession();
        if (!Predict_session.init(msConfig)) {
            Log.e("MS_LITE", "Create Predict_session failed");
            msConfig.free();
        }

        Transform_session = new LiteSession();
        if (!Transform_session.init(msConfig)) {
            Log.e("MS_LITE", "Create Predict_session failed");
            msConfig.free();
        }
        msConfig.free();
        ```

    - 加载模型文件并构建用于推理的计算图

        ```java
        // Complile graph.
        if (!Predict_session.compileGraph(style_predict_model)) {
            Log.e("MS_LITE", "Compile style_predict graph failed");
            style_predict_model.freeBuffer();
        }
        if (!Transform_session.compileGraph(style_transform_model)) {
            Log.e("MS_LITE", "Compile style_transform graph failed");
            style_transform_model.freeBuffer();
        }

        // Note: when use model.freeBuffer(), the model can not be complile graph again.
        style_predict_model.freeBuffer();
        style_transform_model.freeBuffer();
        ```

2. 输入数据： Java目前支持`byte[]`或者`ByteBuffer`两种类型的数据，设置输入Tensor的数据。

   - 在输入数据之前，将float数组转换为byte数组。

       ```java

        public static byte[] floatArrayToByteArray(float[] floats) {
            ByteBuffer buffer = ByteBuffer.allocate(4 * floats.length);
            buffer.order(ByteOrder.nativeOrder());
            FloatBuffer floatBuffer = buffer.asFloatBuffer();
            floatBuffer.put(floats);
            return buffer.array();
        }
        ```

   - 通过`ByteBuffer`输入数据。`contentImage`为用户提供的图片，`styleBitmap`为预置风格图片。

       ```java
        public ModelExecutionResult execute(Bitmap contentImage, Bitmap styleBitmap) {
            Log.i(TAG, "running models");
            fullExecutionTime = SystemClock.uptimeMillis();
            preProcessTime = SystemClock.uptimeMillis();
            ByteBuffer contentArray =
                    ImageUtils.bitmapToByteBuffer(contentImage, CONTENT_IMAGE_SIZE, CONTENT_IMAGE_SIZE, 0, 255);
            ByteBuffer input = ImageUtils.bitmapToByteBuffer(styleBitmap, STYLE_IMAGE_SIZE, STYLE_IMAGE_SIZE, 0, 255);
       ```

3. 对输入Tensor按照模型进行推理，获取输出Tensor，并进行后处理。

   - 使用`runGraph`对预置图片进行模型推理，并获取结果`Predict_results`。

        ```java
        List<MSTensor> Predict_inputs = Predict_session.getInputs();
        if (Predict_inputs.size() != 1) {
            return null;
        }
        MSTensor Predict_inTensor = Predict_inputs.get(0);
        Predict_inTensor.setData(input);

        preProcessTime = SystemClock.uptimeMillis() - preProcessTime;
        stylePredictTime = SystemClock.uptimeMillis();


        if (!Predict_session.runGraph()) {
            Log.e("MS_LITE", "Run Predict_graph failed");
            return null;
        }
        stylePredictTime = SystemClock.uptimeMillis() - stylePredictTime;
        Log.d(TAG, "Style Predict Time to run: " + stylePredictTime);

        // Get output tensor values.
        List<String> tensorNames = Predict_session.getOutputTensorNames();
        Map<String, MSTensor> outputs = Predict_session.getOutputMapByTensor();
        Set<Map.Entry<String, MSTensor>> entrys = outputs.entrySet();

        float[] Predict_results = null;
        for (String tensorName : tensorNames) {
            MSTensor output = outputs.get(tensorName);
            if (output == null) {
                Log.e("MS_LITE", "Can not find Predict_session output " + tensorName);
                return null;
            }
            int type = output.getDataType();
            Predict_results = output.getFloatData();
        }
       ```

   - 利用上一步获取的结果，再次对用户图片进行模型推理，得到风格转换的数据`transform_results`。

        ```java
            List<MSTensor> Transform_inputs = Transform_session.getInputs();
            // transform model have 2 input tensor,  tensor0: 1*1*1*100,   tensor1；1*384*384*3
            MSTensor Transform_inputs_inTensor0 = Transform_inputs.get(0);
            Transform_inputs_inTensor0.setData(floatArrayToByteArray(Predict_results));

            MSTensor Transform_inputs_inTensor1 = Transform_inputs.get(1);
            Transform_inputs_inTensor1.setData(contentArray);


            styleTransferTime = SystemClock.uptimeMillis();

            if (!Transform_session.runGraph()) {
                Log.e("MS_LITE", "Run Transform_graph failed");
                return null;
            }

            styleTransferTime = SystemClock.uptimeMillis() - styleTransferTime;
            Log.d(TAG, "Style apply Time to run: " + styleTransferTime);

            postProcessTime = SystemClock.uptimeMillis();

            // Get output tensor values.
            List<String> Transform_tensorNames = Transform_session.getOutputTensorNames();
            Map<String, MSTensor> Transform_outputs = Transform_session.getOutputMapByTensor();

            float[] transform_results = null;
            for (String tensorName : Transform_tensorNames) {
                MSTensor output1 = Transform_outputs.get(tensorName);
                if (output1 == null) {
                    Log.e("MS_LITE", "Can not find Transform_session output " + tensorName);
                    return null;
                }
                transform_results = output1.getFloatData();
            }
        ```

   - 对输出节点的数据进行处理，得到推理后的最终结果。

        ```java
        float[][][][] outputImage = new float[1][][][];  // 1 384 384 3
        for (int x = 0; x < 1; x++) {
            float[][][] arrayThree = new float[CONTENT_IMAGE_SIZE][][];
            for (int y = 0; y < CONTENT_IMAGE_SIZE; y++) {
                float[][] arrayTwo = new float[CONTENT_IMAGE_SIZE][];
                for (int z = 0; z < CONTENT_IMAGE_SIZE; z++) {
                    float[] arrayOne = new float[3];
                    for (int i = 0; i < 3; i++) {
                        int n = i + z * 3 + y * CONTENT_IMAGE_SIZE * 3 + x * CONTENT_IMAGE_SIZE * CONTENT_IMAGE_SIZE * 3;
                        arrayOne[i] = transform_results[n];
                    }
                    arrayTwo[z] = arrayOne;
                }
                arrayThree[y] = arrayTwo;
            }
            outputImage[x] = arrayThree;
        }


        Bitmap styledImage =
                ImageUtils.convertArrayToBitmap(outputImage, CONTENT_IMAGE_SIZE, CONTENT_IMAGE_SIZE);
        postProcessTime = SystemClock.uptimeMillis() - postProcessTime;

        fullExecutionTime = SystemClock.uptimeMillis() - fullExecutionTime;
        Log.d(TAG, "Time to run everything: $" + fullExecutionTime);

        return new ModelExecutionResult(styledImage,
                preProcessTime,
                stylePredictTime,
                styleTransferTime,
                postProcessTime,
                fullExecutionTime,
                formatExecutionLog());
        ```
