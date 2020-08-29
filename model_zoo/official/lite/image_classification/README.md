## MindSpore Lite 端侧图像分类demo（Android）
  
本示例程序演示了如何在端侧利用MindSpore Lite C++ API（Android JNI）以及MindSpore Lite 图像分类模型完成端侧推理，实现对设备摄像头捕获的内容进行分类，并在App图像预览界面中显示出最可能的分类结果。


### 运行依赖

- Android Studio >= 3.2 (推荐4.0以上版本)
- NDK 21.3
- CMake 3.10
- Android SDK >= 26
- OpenCV >= 4.0.0

### 构建与运行

1. 在Android Studio中加载本示例源码，并安装相应的SDK（指定SDK版本后，由Android Studio自动安装）。 

    ![start_home](images/home.png)

    启动Android Studio后，点击`File->Settings->System Settings->Android SDK`，勾选相应的SDK。如下图所示，勾选后，点击`OK`，Android Studio即可自动安装SDK。

    ![start_sdk](images/sdk_management.png)

    （可选）若安装时出现NDK版本问题，可手动下载相应的[NDK版本](https://developer.android.com/ndk/downloads?hl=zh-cn)（本示例代码使用的NDK版本为21.3），并在`Project Structure`的`Android NDK location`设置中指定SDK的位置。

    ![project_structure](images/project_structure.png)

2. 连接Android设备，运行图像分类应用程序。

    通过USB连接Android设备调试，点击`Run 'app'`即可在您的设备上运行本示例项目。

    * 注：编译过程中Android Studio会自动下载MindSpore Lite、OpenCV、模型文件等相关依赖项，编译过程需做耐心等待。

    ![run_app](images/run_app.PNG)

    Android Studio连接设备调试操作，可参考<https://developer.android.com/studio/run/device?hl=zh-cn>。

3. 在Android设备上，点击“继续安装”，安装完即可查看到设备摄像头捕获的内容和推理结果。

    ![install](images/install.jpg)

    如下图所示，识别出的概率最高的物体是植物。

    ![result](images/app_result.jpg)


## 示例程序详细说明  

本端侧图像分类Android示例程序分为JAVA层和JNI层，其中，JAVA层主要通过Android Camera 2 API实现摄像头获取图像帧，以及相应的图像处理等功能；JNI层在[Runtime](https://www.mindspore.cn/tutorial/zh-CN/master/use/lite_runtime.html)中完成模型推理的过程。

> 此处详细说明示例程序的JNI层实现，JAVA层运用Android Camera 2 API实现开启设备摄像头以及图像帧处理等功能，需读者具备一定的Android开发基础知识。

### 示例程序结构

```
app
|
├── libs # 存放demo jni层依赖的库文件
│   └── arm64-v8a
│       ├── libopencv_java4.so  # opencv
│       ├── libmlkit-label-MS.so  # ndk编译生成的库文件
│       └── libmindspore-lite.so # mindspore lite
|
├── src/main
│   ├── assets # 资源文件
|   |   └── mobilenetv2.ms # 存放模型文件
│   |
│   ├── cpp # 模型加载和预测主要逻辑封装类
|   |   ├── include # 存放MindSpore调用相关的头文件
|   |   |   └── ...
│   |   |
|   |   ├── MindSporeNetnative.cpp # MindSpore调用相关的JNI方法
│   |   └── MindSporeNetnative.h # 头文件
│   |
│   ├── java # java层应用代码
│   │   └── com.huawei.himindsporedemo 
│   │       ├── gallery.classify # 图像处理及MindSpore JNI调用相关实现
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

Android JNI层调用MindSpore C++ API时，需要相关库文件支持。可通过MindSpore Lite[源码编译](https://www.mindspore.cn/lite/docs/zh-CN/master/deploy.html)生成`libmindspore-lite.so`库文件，或直接下载MindSpore Lite提供的已编译完成的AMR64、ARM32、x86等[软件包](#TODO)。

在Android Studio中将编译完成的`libmindspore-lite.so`库文件（可包含多个兼容架构），分别放置在APP工程的`app/libs/arm64-v8a`（ARM64）或`app/libs/armeabi-v7a`（ARM32）目录下，并在应用的`build.gradle`文件中配置CMake编译支持，以及`arm64-v8a`和`armeabi-v7a`的编译支持。

本示例中，build过程由download.gradle文件自动从华为服务器下载libmindspore-lite.so以及OpenCV的libopencv_java4.so库文件，并放置在`app/libs/arm64-v8a`目录下。

* 注：若自动下载失败，请手动下载相关库文件并将其放在对应位置：
* libmindspore-lite.so [下载链接](https://download.mindspore.cn/model_zoo/official/lite/lib/mindspore%20version%200.7/libmindspore-lite.so)
* libopencv_java4.so [下载链接](https://download.mindspore.cn/model_zoo/official/lite/lib/opencv%204.4.0/libopencv_java4.so)

```
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

```
# Set MindSpore Lite Dependencies.
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/include/MindSpore)
add_library(mindspore-lite SHARED IMPORTED )
set_target_properties(mindspore-lite PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/libs/libmindspore-lite.so")

# Set OpenCV Dependecies.
include_directories(${CMAKE_SOURCE_DIR}/opencv/sdk/native/jni/include)
add_library(lib-opencv SHARED IMPORTED )
set_target_properties(lib-opencv PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/libs/libopencv_java4.so")

# Link target library.       
target_link_libraries(
    ...
    mindspore-lite
    lib-opencv
    ...
)
```

### 下载及部署模型文件

从MindSpore Model Hub中下载模型文件，本示例程序中使用的终端图像分类模型文件为`mobilenetv2.ms`，同样通过download.gradle脚本在APP构建时自动下载，并放置在`app/src/main/assets`工程目录下。

* 注：若下载失败请手动下载模型文件，mobilenetv2.ms [下载链接](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2.ms)。


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
        context->thread_num_ = numThread;  //Specify the number of threads to run inference
        
        // Create the mindspore session.
        labelNet->CreateSessionMS(modelBuffer, bufferLen, context);
        delete(context);
        
        ```
        
    - 加载模型文件并构建用于推理的计算图
        ```cpp
        void MSNetWork::CreateSessionMS(char* modelBuffer, size_t bufferLen, mindspore::lite::Context* ctx)
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
    BitmapToMat(env, srcBitmap, matImageSrc);
   // Processing such as zooming the picture size.
    matImgPreprocessed = PreProcessImageData(matImageSrc);  

    ImgDims inputDims; 
    inputDims.channel = matImgPreprocessed.channels();
    inputDims.width = matImgPreprocessed.cols;
    inputDims.height = matImgPreprocessed.rows;
    float *dataHWC = new float[inputDims.channel * inputDims.width * inputDims.height]

    // Copy the image data to be detected to the dataHWC array.
    // The dataHWC[image_size] array here is the intermediate variable of the input MindSpore model tensor.
    float *ptrTmp = reinterpret_cast<float *>(matImgPreprocessed.data);
    for(int i = 0; i < inputDims.channel * inputDims.width * inputDims.height; i++){
       dataHWC[i] = ptrTmp[i];
    }

    // Assign dataHWC[image_size] to the input tensor variable.
    auto msInputs = mSession->GetInputs();
    auto inTensor = msInputs.front();
    memcpy(inTensor->MutableData(), dataHWC,
        inputDims.channel * inputDims.width * inputDims.height * sizeof(float));
    delete[] (dataHWC);
    ```
    
3. 对输入Tensor按照模型进行推理，获取输出Tensor，并进行后处理。    

   - 图执行，端测推理。

        ```cpp
        // After the model and image tensor data is loaded, run inference.
        auto status = mSession->RunGraph();
        ```

   - 获取输出数据。
        ```cpp
        // Get the mindspore inference results.
        auto msOutputs = mSession->GetOutputMapByNode();
        std::string retStr = ProcessRunnetResult(msOutputs);
        ```
        
   - 输出数据的后续处理。
        ```cpp
        std::string ProcessRunnetResult(
                std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> msOutputs){
        
            // Get the branch of the model output.
            // Use iterators to get map elements.
            std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>>::iterator iter;
            iter = msOutputs.begin();

            // The mobilenetv2.ms model output just one branch.
            auto outputString = iter->first;
            auto outputTensor = iter->second;

            float *temp_scores = static_cast<float * >(branch1_tensor[0]->MutableData());
        
            float scores[RET_CATEGORY_SUM];
            for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
                if (temp_scores[i] > 0.5){
                    MS_PRINT("MindSpore scores[%d] : [%f]",  i, temp_scores[i]);
                }
                scores[i] = temp_scores[i];
            }
   
            // Converted to text information that needs to be displayed in the APP. 
            std::string categoryScore = "";
                for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
                    categoryScore += g_labels_name_map[i];
                    categoryScore += ":";
                    std::string score_str = std::to_string(scores[i]);
                    categoryScore += score_str;
                    categoryScore += ";";
                }
                return categoryScore;
        }      
        ```
