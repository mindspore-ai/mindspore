# MindSpore Lite 端侧目标检测demo（Android）

本示例程序演示了如何在端侧利用MindSpore Lite C++ API（Android JNI）以及MindSpore Lite 目标检测模型完成端侧推理，实现对图库或者设备摄像头捕获的内容进行检测，并在App图像预览界面中显示连续目标检测结果。

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

本端侧目标检测Android示例程序分为JAVA层和JNI层，其中，JAVA层主要通过Android Camera 2 API实现摄像头获取图像帧，以及相应的图像处理（针对推理结果画框）等功能；JNI层在[Runtime](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/runtime.html)中完成模型推理的过程。

> 此处详细说明示例程序的JNI层实现，JAVA层运用Android Camera 2 API实现开启设备摄像头以及图像帧处理等功能，需读者具备一定的Android开发基础知识。

### 示例程序结构

```text
app
|
├── libs # 存放demo jni层编译出的库文件
│   └── arm64-v8a
│       │── libmlkit-label-MS.so #
|
├── src/main
│   ├── assets # 资源文件
|   |   └── ssd.ms # 存放模型文件
│   |
│   ├── cpp # 模型加载和预测主要逻辑封装类
|   |   ├── mindspore-lite-x.x.x-mindata-arm64-cpu # minspore源码编译出的调用包,包含demo jni层依赖的库文件及相关的头文件
|   |   |   └── ...
│   |   |
|   |   ├── MindSporeNetnative.cpp # MindSpore调用相关的JNI方法
│   ├── java # java层应用代码
│   │   └── com.huawei.himindsporedemo
│   │       ├── help # 图像处理及MindSpore JNI调用相关实现
│   │       │   └── ...
│   │       └── obejctdetect # 开启摄像头及绘制相关实现
│   │           └── ...
│   │
│   ├── res # 存放Android相关的资源文件
│   └── AndroidManifest.xml # Android配置文件
│
├── CMakeList.txt # cmake编译入口文件
│
├── build.gradle # 其他Android配置文件
├── download.gradle # APP构建时由gradle自动从HuaWei Server下载依赖的库文件及模型文件
└── ...
```

### 配置MindSpore Lite依赖项

Android JNI层调用MindSpore C++ API时，需要相关库文件支持。可通过MindSpore Lite[源码编译](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html)生成`mindspore-lite-{version}-minddata-{os}-{device}.tar.gz`库文件包并解压缩（包含`libmindspore-lite.so`库文件和相关头文件），在本例中需使用生成带图像预处理模块的编译命令。

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
# Set MindSpore Lite Dependencies.
set(MINDSPORELITE_VERSION  mindspore-lite-1.0.1-runtime-arm64-cpu)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION})
add_library(mindspore-lite SHARED IMPORTED )
add_library(minddata-lite SHARED IMPORTED )
set_target_properties(mindspore-lite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/lib/libmindspore-lite.so)
set_target_properties(minddata-lite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/lib/libminddata-lite.so)

# Link target library.
target_link_libraries(
    ...
    mindspore-lite
    minddata-lite
    ...
)
```

### 下载及部署模型文件

从MindSpore Model Hub中下载模型文件，本示例程序中使用的目标检测模型文件为`ssd.ms`，同样通过`download.gradle`脚本在APP构建时自动下载，并放置在`app/src/main/assets`工程目录下。

> 若下载失败请手动下载模型文件，ssd.ms [下载链接](https://download.mindspore.cn/model_zoo/official/lite/ssd_mobilenetv2_lite/ssd.ms)。

### 编写端侧推理代码

在JNI层调用MindSpore Lite C++ API实现端测推理。

推理代码流程如下，完整代码请参见`src/cpp/MindSporeNetnative.cpp`。

1. 加载MindSpore Lite模型文件，构建上下文、会话以及用于推理的计算图。  

    - 加载模型文件：创建并配置用于模型推理的上下文

        ```cpp
        // Buffer is the model data passed in by the Java layer
        jlong bufferLen = env->GetDirectBufferCapacity(buffer);
        char *modelBuffer = CreateLocalModelBuffer(env, buffer);  
        ```

    - 创建会话

        ```cpp
        void **labelEnv = new void *;
        MSNetWork *labelNet = new MSNetWork;
        *labelEnv = labelNet;

        // Create context.
        lite::Context *context = new lite::Context;
        context->cpu_bind_mode_ = lite::NO_BIND;
        context->device_ctx_.type = lite::DT_CPU;
        context->thread_num_ = numThread;  //Specify the number of threads to run inference

        // Create the mindspore session.
        labelNet->CreateSessionMS(modelBuffer, bufferLen, "device label", context);
        delete context;

        ```

    - 加载模型文件并构建用于推理的计算图

        ```cpp
        void MSNetWork::CreateSessionMS(char* modelBuffer, size_t bufferLen, std::string name, mindspore::lite::Context* ctx)
        {
            CreateSession(modelBuffer, bufferLen, ctx);  
            session = mindspore::session::LiteSession::CreateSession(ctx);
            auto model = mindspore::lite::Model::Import(modelBuffer, bufferLen);
            int ret = session->CompileGraph(model); // Compile Graph
        }
        ```

2. 将输入图片转换为传入MindSpore模型的Tensor格式。

    将待检测图片数据转换为输入MindSpore模型的Tensor。

    ```cpp
    // Convert the Bitmap image passed in from the JAVA layer to Mat for OpenCV processing
        LiteMat lite_mat_bgr,lite_norm_mat_cut;

       if (!BitmapToLiteMat(env, srcBitmap, lite_mat_bgr)){
        MS_PRINT("BitmapToLiteMat error");
           return NULL;
       }
       int srcImageWidth = lite_mat_bgr.width_;
       int srcImageHeight = lite_mat_bgr.height_;
       if(!PreProcessImageData(lite_mat_bgr, lite_norm_mat_cut)){
        MS_PRINT("PreProcessImageData error");
           return NULL;
       }
       ImgDims inputDims;
       inputDims.channel =lite_norm_mat_cut.channel_;
       inputDims.width = lite_norm_mat_cut.width_;
       inputDims.height = lite_norm_mat_cut.height_;

       // Get the mindsore inference environment which created in loadModel().
       void **labelEnv = reinterpret_cast<void **>(netEnv);
       if (labelEnv == nullptr) {
           MS_PRINT("MindSpore error, labelEnv is a nullptr.");
           return NULL;
       }
       MSNetWork *labelNet = static_cast<MSNetWork *>(*labelEnv);

       auto mSession = labelNet->session;
       if (mSession == nullptr) {
           MS_PRINT("MindSpore error, Session is a nullptr.");
           return NULL;
       }
       MS_PRINT("MindSpore get session.");

       auto msInputs = mSession->GetInputs();
       auto inTensor = msInputs.front();

       float *dataHWC = reinterpret_cast<float *>(lite_norm_mat_cut.data_ptr_);
       // copy input Tensor
       memcpy(inTensor->MutableData(), dataHWC,
              inputDims.channel * inputDims.width * inputDims.height * sizeof(float));
       delete[] (dataHWC);
   ```

3. 进行模型推理前，输入tensor格式为 NHWC，shape为1:300:300:3，格式为RGB,  并对输入tensor做标准化处理.

   ```cpp
    bool PreProcessImageData(LiteMat &lite_mat_bgr,LiteMat &lite_norm_mat_cut) {
       bool ret=false;
       LiteMat lite_mat_resize;
       ret = ResizeBilinear(lite_mat_bgr, lite_mat_resize, 300, 300);
       if (!ret) {
           MS_PRINT("ResizeBilinear error");
           return false;
       }
       LiteMat lite_mat_convert_float;
       ret = ConvertTo(lite_mat_resize, lite_mat_convert_float, 1.0 / 255.0);
       if (!ret) {
           MS_PRINT("ConvertTo error");
           return false;
       }

       float means[3] = {0.485, 0.456, 0.406};
       float vars[3] = {1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225};
       SubStractMeanNormalize(lite_mat_convert_float, lite_norm_mat_cut, means, vars);
       return true;
   }
   ```

4. 对输入Tensor按照模型进行推理，获取输出Tensor，并进行后处理。

   - 图执行，端测推理。

        ```cpp
        // After the model and image tensor data is loaded, run inference.
        auto status = mSession->RunGraph();
        ```

   - 获取输出数据。

        ```cpp
        auto names = mSession->GetOutputTensorNames();
            typedef std::unordered_map<std::string,
                    std::vector<mindspore::tensor::MSTensor *>> Msout;
        std::unordered_map<std::string,
                    mindspore::tensor::MSTensor *> msOutputs;
        for (const auto &name : names) {
                auto temp_dat =mSession->GetOutputByTensorName(name);  
                msOutputs.insert(std::pair<std::string, mindspore::tensor::MSTensor *> {name, temp_dat});  
            }
        std::string retStr = ProcessRunnetResult(msOutputs, ret);
        ```

   - 模型有2个输出，输出1是目标的类别置信度，维度为1：1917: 81； 输出2是目标的矩形框坐标偏移量，维度为1:1917:4。 为了得出目标的实际矩形框，需要根据偏移量计算出矩形框的位置。这部分在 getDefaultBoxes中实现。

        ```cpp
        void SSDModelUtil::getDefaultBoxes() {
            float fk[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            std::vector<struct WHBox> all_sizes;
            struct Product mProductData[19 * 19] = {0};

            for (int i = 0; i < 6; i++) {
                fk[i] = config.model_input_height / config.steps[i];
            }
            float scale_rate =
                    (config.max_scale - config.min_scale) / (sizeof(config.num_default) / sizeof(int) - 1);
            float scales[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
            for (int i = 0; i < sizeof(config.num_default) / sizeof(int); i++) {
                scales[i] = config.min_scale + scale_rate * i;
            }

            for (int idex = 0; idex < sizeof(config.feature_size) / sizeof(int); idex++) {
                float sk1 = scales[idex];
                float sk2 = scales[idex + 1];
                float sk3 = sqrt(sk1 * sk2);
                struct WHBox tempWHBox;

                all_sizes.clear();

                if (idex == 0) {
                    float w = sk1 * sqrt(2);
                    float h = sk1 / sqrt(2);

                    tempWHBox.boxw = 0.1;
                    tempWHBox.boxh = 0.1;
                    all_sizes.push_back(tempWHBox);

                    tempWHBox.boxw = w;
                    tempWHBox.boxh = h;
                    all_sizes.push_back(tempWHBox);

                    tempWHBox.boxw = h;
                    tempWHBox.boxh = w;
                    all_sizes.push_back(tempWHBox);

                } else {
                    tempWHBox.boxw = sk1;
                    tempWHBox.boxh = sk1;
                    all_sizes.push_back(tempWHBox);

                    for (int j = 0; j < sizeof(config.aspect_ratios[idex]) / sizeof(int); j++) {
                        float w = sk1 * sqrt(config.aspect_ratios[idex][j]);
                        float h = sk1 / sqrt(config.aspect_ratios[idex][j]);
                        tempWHBox.boxw = w;
                        tempWHBox.boxh = h;
                        all_sizes.push_back(tempWHBox);
                        tempWHBox.boxw = h;
                        tempWHBox.boxh = w;
                        all_sizes.push_back(tempWHBox);
                    }

                    tempWHBox.boxw = sk3;
                    tempWHBox.boxh = sk3;
                    all_sizes.push_back(tempWHBox);
                }

                for (int i = 0; i < config.feature_size[idex]; i++) {
                    for (int j = 0; j < config.feature_size[idex]; j++) {
                        mProductData[i * config.feature_size[idex] + j].x = i;
                        mProductData[i * config.feature_size[idex] + j].y = j;
                    }
                }

                int productLen = config.feature_size[idex] * config.feature_size[idex];

                for (int i = 0; i < productLen; i++) {
                    for (int j = 0; j < all_sizes.size(); j++) {
                        struct NormalBox tempBox;
                        float cx = (mProductData[i].y + 0.5) / fk[idex];
                        float cy = (mProductData[i].x + 0.5) / fk[idex];
                        tempBox.y = cy;
                        tempBox.x = cx;
                     tempBox.h = all_sizes[j].boxh;
                        tempBox.w = all_sizes[j].boxw;
                     mDefaultBoxes.push_back(tempBox);
                    }
                }
            }
        }
        ```

   - 通过最大值抑制将目标类型置信度较高的输出筛选出来。

        ```cpp
        void SSDModelUtil::nonMaximumSuppression(const YXBoxes *const decoded_boxes,
                                                 const float *const scores,
                                                 const std::vector<int> &in_indexes,
                                                 std::vector<int> &out_indexes, const float nmsThreshold,
                                                 const int count, const int max_results) {
            int nR = 0; //number of results
            std::vector<bool> del(count, false);
            for (size_t i = 0; i < in_indexes.size(); i++) {
                if (!del[in_indexes[i]]) {
                    out_indexes.push_back(in_indexes[i]);
                    if (++nR == max_results) {
                        break;
                    }
                    for (size_t j = i + 1; j < in_indexes.size(); j++) {
                        const auto boxi = decoded_boxes[in_indexes[i]], boxj = decoded_boxes[in_indexes[j]];
                        float a[4] = {boxi.xmin, boxi.ymin, boxi.xmax, boxi.ymax};
                        float b[4] = {boxj.xmin, boxj.ymin, boxj.xmax, boxj.ymax};
                     if (IOU(a, b) > nmsThreshold) {
                            del[in_indexes[j]] = true;
                        }
                    }
                }
            }
        }
        ```

   - 对每类的概率大于阈值，通过NMS算法筛选出矩形框后， 还需要将输出的矩形框恢复到原图尺寸。

        ```cpp
        std::string SSDModelUtil::getDecodeResult(float *branchScores, float *branchBoxData) {
            std::string result = "";
            NormalBox tmpBox[1917] = {0};
            float mScores[1917][81] = {0};
            float outBuff[1917][7] = {0};
            float scoreWithOneClass[1917] = {0};
            int outBoxNum = 0;
            YXBoxes decodedBoxes[1917] = {0};

            // Copy branch outputs box data to tmpBox.
            for (int i = 0; i < 1917; ++i) {
                tmpBox[i].y = branchBoxData[i * 4 + 0];
                tmpBox[i].x = branchBoxData[i * 4 + 1];
                tmpBox[i].h = branchBoxData[i * 4 + 2];
                tmpBox[i].w = branchBoxData[i * 4 + 3];
            }

            // Copy branch outputs score to mScores.
            for (int i = 0; i < 1917; ++i) {
                for (int j = 0; j < 81; ++j) {
                    mScores[i][j] = branchScores[i * 81 + j];
                }
            }

             ssd_boxes_decode(tmpBox, decodedBoxes);
             const float nms_threshold = 0.3;
             for (int i = 1; i < 81; i++) {
                 std::vector<int> in_indexes;
                 for (int j = 0; j < 1917; j++) {
                     scoreWithOneClass[j] = mScores[j][i];
                  //   if (mScores[j][i] > 0.1) {
                     if (mScores[j][i] > g_thres_map[i]) {
                         in_indexes.push_back(j);
                     }
                 }
                 if (in_indexes.size() == 0) {
                     continue;
                 }

                 sort(in_indexes.begin(), in_indexes.end(),
                      [&](int a, int b) { return scoreWithOneClass[a] > scoreWithOneClass[b]; });
                 std::vector<int> out_indexes;

                 nonMaximumSuppression(decodedBoxes, scoreWithOneClass, in_indexes, out_indexes,
                                       nms_threshold);
                 for (int k = 0; k < out_indexes.size(); k++) {
                     outBuff[outBoxNum][0] = out_indexes[k]; //image id
                     outBuff[outBoxNum][1] = i; //labelid
                     outBuff[outBoxNum][2] = scoreWithOneClass[out_indexes[k]]; //scores
                     outBuff[outBoxNum][3] =
                             decodedBoxes[out_indexes[k]].xmin * inputImageWidth / 300;
                     outBuff[outBoxNum][4] =
                             decodedBoxes[out_indexes[k]].ymin * inputImageHeight / 300;
                     outBuff[outBoxNum][5] =
                             decodedBoxes[out_indexes[k]].xmax * inputImageWidth / 300;
                     outBuff[outBoxNum][6] =
                             decodedBoxes[out_indexes[k]].ymax * inputImageHeight / 300;
                     outBoxNum++;
                 }
             }
             MS_PRINT("outBoxNum %d", outBoxNum);

             for (int i = 0; i < outBoxNum; ++i) {
                 std::string tmpid_str = std::to_string(outBuff[i][0]);
                 result += tmpid_str; // image ID
                 result += "_";
                 // tmpid_str = std::to_string(outBuff[i][1]);
                 MS_PRINT("label_classes i %d, outBuff %d",i, (int) outBuff[i][1]);
                 tmpid_str = label_classes[(int) outBuff[i][1]];
                 result += tmpid_str; // label id
                 result += "_";
                 tmpid_str = std::to_string(outBuff[i][2]);
                 result += tmpid_str; // scores
                 result += "_";
                 tmpid_str = std::to_string(outBuff[i][3]);
                 result += tmpid_str; // xmin
                 result += "_";
                 tmpid_str = std::to_string(outBuff[i][4]);
                 result += tmpid_str; // ymin
                 result += "_";
                 tmpid_str = std::to_string(outBuff[i][5]);
                 result += tmpid_str; // xmax
                 result += "_";
                 tmpid_str = std::to_string(outBuff[i][6]);
                 result += tmpid_str; // ymax
                 result += ";";
             }
            return result;
        }
        ```
