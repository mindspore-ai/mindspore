/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <jni.h>
#include <android/bitmap.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <utility>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include "include/errorcode.h"
#include "include/ms_tensor.h"
#include "ImageMindSporeNetnative.h"
#include "MSNetWork.h"
#include "lite_cv/lite_mat.h"
#include "lite_cv/image_process.h"

using mindspore::dataset::LiteMat;
using mindspore::dataset::LPixelType;
using mindspore::dataset::LDataType;
#define MS_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MSJNI", format, ##__VA_ARGS__)


static const int RET_CATEGORY_SUM = 410;

static float g_thres_map[RET_CATEGORY_SUM] = {
        0.23, 0.03, 0.10, 0.13, 0.03,
        0.10, 0.06, 0.09, 0.09, 0.05,
        0.01, 0.04, 0.01, 0.27, 0.05,
        0.16, 0.01, 0.16, 0.04, 0.13,
        0.09, 0.18, 0.10, 0.65, 0.08,
        0.04, 0.08, 0.01, 0.05, 0.20,
        0.01, 0.16, 0.10, 0.10, 0.10,
        0.02, 0.24, 0.08, 0.10, 0.53,
        0.07, 0.05, 0.07, 0.27, 0.02,
        0.01, 0.71, 0.01, 0.06, 0.06,
        0.03, 0.96, 0.03, 0.94, 0.05,
        0.03, 0.14, 0.09, 0.03, 0.11,
        0.50, 0.16, 0.07, 0.07, 0.06,
        0.07, 0.08, 0.10, 0.29, 0.03,
        0.05, 0.11, 0.03, 0.03, 0.03,
        0.01, 0.11, 0.07, 0.03, 0.49,
        0.12, 0.30, 0.10, 0.15, 0.02,
        0.06, 0.17, 0.01, 0.04, 0.07,
        0.06, 0.02, 0.19, 0.20, 0.14,
        0.35, 0.15, 0.01, 0.10, 0.13,
        0.43, 0.11, 0.12, 0.32, 0.01,
        0.22, 0.51, 0.02, 0.04, 0.14,
        0.04, 0.35, 0.35, 0.01, 0.54,
        0.04, 0.02, 0.03, 0.02, 0.38,
        0.13, 0.19, 0.06, 0.01, 0.02,
        0.06, 0.03, 0.04, 0.01, 0.10,
        0.01, 0.07, 0.07, 0.07, 0.33,
        0.08, 0.04, 0.06, 0.07, 0.07,
        0.11, 0.02, 0.32, 0.48, 0.14,
        0.01, 0.01, 0.04, 0.05, 0.04,
        0.16, 0.50, 0.11, 0.03, 0.04,
        0.02, 0.55, 0.17, 0.13, 0.84,
        0.18, 0.03, 0.16, 0.02, 0.06,
        0.03, 0.11, 0.96, 0.36, 0.68,
        0.02, 0.08, 0.02, 0.01, 0.03,
        0.05, 0.14, 0.09, 0.06, 0.03,
        0.20, 0.15, 0.62, 0.03, 0.10,
        0.08, 0.02, 0.02, 0.06, 0.03,
        0.04, 0.01, 0.10, 0.05, 0.04,
        0.02, 0.07, 0.03, 0.32, 0.11,
        0.03, 0.02, 0.03, 0.01, 0.03,
        0.03, 0.25, 0.20, 0.19, 0.03,
        0.11, 0.03, 0.02, 0.03, 0.15,
        0.14, 0.06, 0.11, 0.03, 0.02,
        0.02, 0.52, 0.03, 0.02, 0.02,
        0.02, 0.09, 0.56, 0.01, 0.22,
        0.01, 0.48, 0.14, 0.10, 0.08,
        0.73, 0.39, 0.09, 0.10, 0.85,
        0.31, 0.03, 0.05, 0.01, 0.01,
        0.01, 0.10, 0.28, 0.02, 0.03,
        0.04, 0.03, 0.07, 0.14, 0.20,
        0.10, 0.01, 0.05, 0.37, 0.12,
        0.04, 0.44, 0.04, 0.26, 0.08,
        0.07, 0.27, 0.10, 0.03, 0.01,
        0.03, 0.16, 0.41, 0.16, 0.34,
        0.04, 0.30, 0.04, 0.05, 0.18,
        0.33, 0.03, 0.21, 0.03, 0.04,
        0.22, 0.01, 0.04, 0.02, 0.01,
        0.06, 0.02, 0.08, 0.87, 0.11,
        0.15, 0.05, 0.14, 0.09, 0.08,
        0.22, 0.09, 0.07, 0.06, 0.06,
        0.05, 0.43, 0.70, 0.03, 0.07,
        0.06, 0.07, 0.14, 0.04, 0.01,
        0.03, 0.05, 0.65, 0.06, 0.04,
        0.23, 0.06, 0.75, 0.10, 0.01,
        0.63, 0.41, 0.09, 0.01, 0.01,
        0.18, 0.10, 0.03, 0.01, 0.05,
        0.13, 0.18, 0.03, 0.23, 0.01,
        0.04, 0.03, 0.38, 0.90, 0.21,
        0.18, 0.10, 0.48, 0.08, 0.46,
        0.03, 0.01, 0.02, 0.03, 0.10,
        0.01, 0.09, 0.01, 0.01, 0.01,
        0.10, 0.41, 0.01, 0.06, 0.75,
        0.08, 0.01, 0.01, 0.08, 0.21,
        0.06, 0.02, 0.05, 0.02, 0.05,
        0.09, 0.12, 0.03, 0.06, 0.11,
        0.03, 0.01, 0.01, 0.06, 0.84,
        0.04, 0.81, 0.39, 0.02, 0.29,
        0.77, 0.07, 0.06, 0.22, 0.23,
        0.23, 0.01, 0.02, 0.13, 0.04,
        0.19, 0.04, 0.08, 0.27, 0.09,
        0.06, 0.01, 0.03, 0.21, 0.04,
};

char *ImageCreateLocalModelBuffer(JNIEnv *env, jobject modelBuffer) {
    jbyte *modelAddr = static_cast<jbyte *>(env->GetDirectBufferAddress(modelBuffer));
    int modelLen = static_cast<int>(env->GetDirectBufferCapacity(modelBuffer));
    char *buffer(new char[modelLen]);
    memcpy(buffer, modelAddr, modelLen);
    return buffer;
}

/**
 * To process the result of mindspore inference.
 * @param msOutputs
 * @return
 */
std::string ImageProcessRunnetResult(const int RET_CATEGORY_SUM,
                                std::unordered_map<std::string, mindspore::tensor::MSTensor *> msOutputs) {
    // Get the branch of the model output.
    // Use iterators to get map elements.
    std::unordered_map<std::string, mindspore::tensor::MSTensor *>::iterator iter;
    iter = msOutputs.begin();

    // The mobilenetv2.ms model output just one branch.
    auto outputTensor = iter->second;

    int tensorNum = outputTensor->ElementsNum();
    MS_PRINT("Number of tensor elements:%d", tensorNum);

    // Get a pointer to the first score.
    float *temp_scores = static_cast<float *>(outputTensor->MutableData());
    float scores[RET_CATEGORY_SUM];
    for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
        scores[i] = temp_scores[i];
    }

    const float unifiedThre = 0.5;
    const float probMax = 1.0;
    for (size_t i = 0; i < RET_CATEGORY_SUM; ++i) {
        float threshold = g_thres_map[i];
        float tmpProb = scores[i];
        if (tmpProb < threshold) {
            tmpProb = tmpProb / threshold * unifiedThre;
        } else {
            tmpProb = (tmpProb - threshold) / (probMax - threshold) * unifiedThre + unifiedThre;
        }
        scores[i] = tmpProb;
    }

    // Score for each category.
    // Converted to text information that needs to be displayed in the APP.
    std::string categoryScore = "";
    for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
        categoryScore += std::to_string(i);
        categoryScore += ":";
        std::string score_str = std::to_string(scores[i]);
        categoryScore += score_str;
        categoryScore += ";";
    }
    MS_PRINT("categoryScore:%s", categoryScore.c_str());

    return categoryScore;
}

bool ImageBitmapToLiteMat(JNIEnv *env, const jobject &srcBitmap, LiteMat *lite_mat) {
    bool ret = false;
    AndroidBitmapInfo info;
    void *pixels = nullptr;
    LiteMat &lite_mat_bgr = *lite_mat;
    AndroidBitmap_getInfo(env, srcBitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        MS_PRINT("Image Err, Request RGBA");
        return false;
    }
    AndroidBitmap_lockPixels(env, srcBitmap, &pixels);
    if (info.stride == info.width * 4) {
        ret = InitFromPixel(reinterpret_cast<const unsigned char *>(pixels),
                            LPixelType::RGBA2RGB, LDataType::UINT8,
                            info.width, info.height, lite_mat_bgr);
        if (!ret) {
            MS_PRINT("Init From RGBA error");
        }
    } else {
        unsigned char *pixels_ptr = new unsigned char[info.width * info.height * 4];
        unsigned char *ptr = pixels_ptr;
        unsigned char *data = reinterpret_cast<unsigned char *>(pixels);
        for (int i = 0; i < info.height; i++) {
            memcpy(ptr, data, info.width * 4);
            ptr += info.width * 4;
            data += info.stride;
        }
        ret = InitFromPixel(reinterpret_cast<const unsigned char *>(pixels_ptr),
                            LPixelType::RGBA2RGB, LDataType::UINT8,
                            info.width, info.height, lite_mat_bgr);
        if (!ret) {
            MS_PRINT("Init From RGBA error");
        }
        delete[] (pixels_ptr);
    }
    AndroidBitmap_unlockPixels(env, srcBitmap);
    return ret;
}

bool ImagePreProcessImageData(const LiteMat &lite_mat_bgr, LiteMat *lite_norm_mat_ptr) {
    bool ret = false;
    LiteMat lite_mat_resize;
    LiteMat &lite_norm_mat_cut = *lite_norm_mat_ptr;
    ret = ResizeBilinear(lite_mat_bgr, lite_mat_resize, 256, 256);
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
    LiteMat lite_mat_cut;
    ret = Crop(lite_mat_convert_float, lite_mat_cut, 16, 16, 224, 224);
    if (!ret) {
        MS_PRINT("Crop error");
        return false;
    }
    std::vector<float> means = {0.485, 0.456, 0.406};
    std::vector<float> stds = {0.229, 0.224, 0.225};
    SubStractMeanNormalize(lite_mat_cut, lite_norm_mat_cut, means, stds);
    return true;
}


/**
 * The Java layer reads the model into MappedByteBuffer or ByteBuffer to load the model.
 */
extern "C"
JNIEXPORT jlong JNICALL
Java_com_mindspore_classificationforpet_gallery_classify_ImageTrackingMobile_loadModel(JNIEnv *env,
                                                                                      jobject thiz,
                                                                                      jobject model_buffer,
                                                                                      jint num_thread) {
    if (nullptr == model_buffer) {
        MS_PRINT("error, buffer is nullptr!");
        return (jlong) nullptr;
    }
    jlong bufferLen = env->GetDirectBufferCapacity(model_buffer);
    if (0 == bufferLen) {
        MS_PRINT("error, bufferLen is 0!");
        return (jlong) nullptr;
    }

    char *modelBuffer = ImageCreateLocalModelBuffer(env, model_buffer);
    if (modelBuffer == nullptr) {
        MS_PRINT("modelBuffer create failed!");
        return (jlong) nullptr;
    }

    // To create a mindspore network inference environment.
    void **labelEnv = new void *;
    MSNetWork *labelNet = new MSNetWork;
    *labelEnv = labelNet;

    mindspore::lite::Context *context = new mindspore::lite::Context;
    context->thread_num_ = num_thread;

    labelNet->CreateSessionMS(modelBuffer, bufferLen, context);
    delete context;

    if (labelNet->session() == nullptr) {
        MS_PRINT("MindSpore create session failed!.");
        delete labelNet;
        delete labelEnv;
        return (jlong) nullptr;
    }

    if (model_buffer != nullptr) {
        env->DeleteLocalRef(model_buffer);
    }

    return (jlong) labelEnv;
}

/**
 * After the inference environment is successfully created,
 * sending a picture to the model and run inference.
 */
extern "C" JNIEXPORT jstring JNICALL
Java_com_mindspore_classificationforpet_gallery_classify_ImageTrackingMobile_runNet(JNIEnv *env,
                                                                                   jclass type,
                                                                                   jlong netEnv,
                                                                                   jobject srcBitmap) {
    LiteMat lite_mat_bgr, lite_norm_mat_cut;

    if (!ImageBitmapToLiteMat(env, srcBitmap, &lite_mat_bgr)) {
        MS_PRINT("ImageBitmapToLiteMat error");
        return NULL;
    }
    if (!ImagePreProcessImageData(lite_mat_bgr, &lite_norm_mat_cut)) {
        MS_PRINT("ImagePreProcessImageData error");
        return NULL;
    }

    ImgDims inputDims;
    inputDims.channel = lite_norm_mat_cut.channel_;
    inputDims.width = lite_norm_mat_cut.width_;
    inputDims.height = lite_norm_mat_cut.height_;

    // Get the mindsore inference environment which created in loadModel().
    void **labelEnv = reinterpret_cast<void **>(netEnv);
    if (labelEnv == nullptr) {
        MS_PRINT("MindSpore error, labelEnv is a nullptr.");
        return NULL;
    }
    MSNetWork *labelNet = static_cast<MSNetWork *>(*labelEnv);

    auto mSession = labelNet->session();
    if (mSession == nullptr) {
        MS_PRINT("MindSpore error, Session is a nullptr.");
        return NULL;
    }
    MS_PRINT("MindSpore get session.");

    auto msInputs = mSession->GetInputs();
    if (msInputs.size() == 0) {
        MS_PRINT("MindSpore error, msInputs.size() equals 0.");
        return NULL;
    }
    auto inTensor = msInputs.front();

    float *dataHWC = reinterpret_cast<float *>(lite_norm_mat_cut.data_ptr_);
    // Copy dataHWC to the model input tensor.
    memcpy(inTensor->MutableData(), dataHWC,
           inputDims.channel * inputDims.width * inputDims.height * sizeof(float));

    // After the model and image tensor data is loaded, run inference.
    auto status = mSession->RunGraph();

    if (status != mindspore::lite::RET_OK) {
        MS_PRINT("MindSpore run net error.");
        return NULL;
    }

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

    std::string resultStr = ImageProcessRunnetResult(::RET_CATEGORY_SUM, msOutputs);
    MS_PRINT("resultStr:%s", resultStr.c_str());

    const char *resultCharData = resultStr.c_str();

    return (env)->NewStringUTF(resultCharData);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mindspore_classificationforpet_gallery_classify_ImageTrackingMobile_unloadModel(JNIEnv *env,
                                                                                        jclass type,
                                                                                        jlong netEnv) {
    MS_PRINT("MindSpore release net.");
    void **labelEnv = reinterpret_cast<void **>(netEnv);
    if (labelEnv == nullptr) {
        MS_PRINT("MindSpore error, labelEnv is a nullptr.");
    }
    MSNetWork *labelNet = static_cast<MSNetWork *>(*labelEnv);

    labelNet->ReleaseNets();

    return (jboolean) true;
}
