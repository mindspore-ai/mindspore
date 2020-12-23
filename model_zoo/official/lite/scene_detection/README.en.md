# MindSpore Lite Scene Detection Demo (Android)

This sample application demonstrates how to use the MindSpore Lite C++ API (Android JNI) and MindSpore Lite scene detection model to perform inference on the device, detect the content captured by the device camera, and display the continuous objective detection result on the image preview screen of the app.

## Running Dependencies

- Android Studio 3.2 or later (Android 4.0 or later is recommended.)
- NDK 21.3
- CMake 3.10
- Android software development kit (SDK) 26 or later

## Building and Running

1. Load the sample source code to Android Studio and install the corresponding SDK. (After the SDK version is specified, Android Studio automatically installs the SDK.)

    ![start_home](images/home.png)

    Start Android Studio, click `File > Settings > System Settings > Android SDK`, and select the corresponding SDK. As shown in the following figure, select an SDK and click `OK`. Android Studio automatically installs the SDK.

    ![start_sdk](images/sdk_management.png)

    If an Android Studio configuration error occurs, solve it by referring to the following table.

    | | Error | Solution |
    | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | 1    | Gradle sync failed: NDK not configured.                      | Specify the NDK installation directory in the local.properties file: ndk.dir={NDK installation directory} |
    | 2    | Requested NDK version did not match the version requested by ndk.dir | Manually download the corresponding [NDK version](https://developer.android.com/ndk/downloads) and specify the SDK location in the `Android NDK location` field (see the following figure). |
    | 3    | This version of Android Studio cannot open this project, please retry with Android Studio or newer. | Choose `Help` > `Checkout for Updates` on the toolbar to update the version. |
    | 4    | SSL peer shut down incorrectly                               | Rebuild. |

    ![project_structure](images/project_structure.png)

2. Connect to an Android device and run the scene detection sample application.

    Connect to the Android device through a USB cable for debugging. Click `Run 'app'` to run the sample project on your device.
   > During the building, Android Studio automatically downloads dependencies related to MindSpore Lite and model files. Please wait.

    ![run_app](images/run_app.PNG)

    For details about how to connect the Android Studio to a device for debugging, see <https://developer.android.com/studio/run/device>.

3. Continue the installation on the Android device. After the installation is complete, you can view the content captured by a camera and the inference result.

    ![install](images/install.jpg)

## Detailed Description of the Sample Application  

The scene detection sample application on the Android device includes a Java layer and a JNI layer. At the Java layer, the Android Camera 2 API is used to enable a camera to obtain image frames and process images (drawing frames based on the inference result). At the JNI layer, the model inference process is completed in [runtime](https://www.mindspore.cn/tutorial/lite/en/master/use/runtime.html).

> This following describes the JNI layer implementation of the sample application. At the Java layer, the Android Camera 2 API is used to enable a device camera and process image frames. Readers are expected to have the basic Android development knowledge.

### Sample Application Structure

```text
app
|
├── libs # Library files built by the demo JNI layer
│   └── arm64-v8a
│       │── libmlkit-label-MS.so #
|
├── src/main
│   ├── assets # Resource file
|   |   └── mobilenetv2.ms # Model file
│   |
│   ├── cpp # Main logic encapsulation classes for model loading and prediction
|   |   ├── mindspore-lite-x.x.x-mindata-arm64-cpu # Calling package built from the MindSpore source code, including the library files and related header files on which the demo JNI layer depends
|   |   |   └── ...
│   |   |
|   |   ├── MindSporeNetnative.cpp # JNI methods related to MindSpore calling
│   ├── java # Application code at the Java layer
│   │   └── com.huawei.himindsporedemo
│   │       ├── help # Implementation related to image processing and MindSpore JNI calling
│   │       │   └── ...
│   │       └── obejctdetect # Implementation related to camera enabling and drawing
│   │           └── ...
│   │
│   ├── res # Resource files related to Android
│   └── AndroidManifest.xml # Android configuration file
│
├── CMakeLists.txt # CMake compilation entry file
│
├── build.gradle # Other Android configuration file
├── download.gradle # During app building, the .gradle file automatically downloads the dependent library files and model files from the Huawei server.
└── ...
```

### Configuring MindSpore Lite Dependencies

When MindSpore C++ APIs are called at the Android JNI layer, related library files are required. You can refer to [Building MindSpore Lite](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html) to generate the `mindspore-lite-{version}-minddata-{os}-{device}.tar.gz` library file package (including the `libmindspore-lite.so` library file and related header files) and decompress it. The following example uses the build command with the image preprocessing module.

> version: version number in the output file, which is the same as the version number of the built branch code.
>
> device: The value can be cpu (built-in CPU operator) or gpu (built-in CPU and GPU operator).
>
> os: operating system to be deployed in the output file.

In this example, the MindSpore Lite version file is automatically downloaded by the download.gradle file during the build process and stored in the `app/src/main/cpp/` directory.

> If the automatic download fails, manually download the library file [mindspore-lite-1.0.1-runtime-arm64-cpu.tar.gz](https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.0.1/lite/android_aarch64/mindspore-lite-1.0.1-runtime-arm64-cpu.tar.gz), and decompress and save it to the directory.

In the `build.gradle` file of the app, configure the build support of both CMake and `arm64-v8a`:

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

Create a link to the `.so` library file in the `app/CMakeLists.txt` file:

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

### Downloading and Deploying the Model File

Download the model file from MindSpore Model Hub. The scene detection model file used in this sample application is `mobilenetv2.ms`, which is automatically downloaded during app building using the `download.gradle` script and stored in the `app/src/main/assets` project directory.

> If the download fails, manually download the model file [mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2.ms).

### Writing On-Device Inference Code

Call MindSpore Lite C++ APIs at the JNI layer to implement on-device inference.

The inference code process is as follows. For details about the complete code, see `src/cpp/MindSporeNetnative.cpp`.

1. Load the MindSpore Lite model file and build the context, session, and computational graph for inference.  

    - Load a model file.

        ```cpp
        jlong bufferLen = env->GetDirectBufferCapacity(model_buffer);
        if (0 == bufferLen) {
            MS_PRINT("error, bufferLen is 0!");
            return (jlong) nullptr;
        }

        char *modelBuffer = CreateLocalModelBuffer(env, model_buffer);
        if (modelBuffer == nullptr) {
            MS_PRINT("modelBuffer create failed!");
            return (jlong) nullptr;
        }
        ```

    - Create a session.

        ```cpp
        void **labelEnv = new void *;
        MSNetWork *labelNet = new MSNetWork;
        *labelEnv = labelNet;

        mindspore::lite::Context *context = new mindspore::lite::Context;
        context->thread_num_ = num_thread;
        context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = mindspore::lite::NO_BIND;
        context->device_list_[0].device_info_.cpu_device_info_.enable_float16_ = false;
        context->device_list_[0].device_type_ = mindspore::lite::DT_CPU;

        labelNet->CreateSessionMS(modelBuffer, bufferLen, context);
        delete context;
        ```

    - Load the model file and build a computational graph for inference.

        ```cpp
        void
        MSNetWork::CreateSessionMS(char *modelBuffer, size_t bufferLen, mindspore::lite::Context *ctx) {
            session_ = mindspore::session::LiteSession::CreateSession(ctx);
            if (session_ == nullptr) {
                MS_PRINT("Create Session failed.");
                return;
            }

            // Compile model.
            model_ = mindspore::lite::Model::Import(modelBuffer, bufferLen);
            if (model_ == nullptr) {
                ReleaseNets();
                MS_PRINT("Import model failed.");
                return;
            }

            int ret = session_->CompileGraph(model_);
            if (ret != mindspore::lite::RET_OK) {
                ReleaseNets();
                MS_PRINT("CompileGraph failed.");
                return;
            }
        }
        ```

2. Convert the input image into the Tensor format of the MindSpore model.

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

3. Perform inference on the input tensor based on the model to obtain the output tensor.

   - Perform graph execution and on-device inference.

        ```cpp
        // After the model and image tensor data is loaded, run inference.
        auto status = mSession->RunGraph();

        if (status != mindspore::lite::RET_OK) {
            MS_PRINT("MindSpore run net error.");
            return NULL;
        }
        ```

   - Obtain the output data.

        ```cpp
        /**
         * Get the mindspore inference results.
         * Return the map of output node name and MindSpore Lite MSTensor.
         */
        auto names = mSession->GetOutputTensorNames();
        std::unordered_map<std::string, mindspore::tensor::MSTensor *> msOutputs;
        for (const auto &name : names) {
            auto temp_dat = mSession->GetOutputByTensorName(name);
            msOutputs.insert(std::pair<std::string, mindspore::tensor::MSTensor *>{name, temp_dat});
        }
        ```
