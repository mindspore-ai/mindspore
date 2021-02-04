/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <sstream>
#include <cstring>
#include <set>
#include <utility>
#include "include/errorcode.h"
#include "include/ms_tensor.h"
#include "MSNetWork.h"
#include "ssd_util/ssd_util.h"
#include "lite_cv/lite_mat.h"
#include "lite_cv/image_process.h"

using mindspore::dataset::LiteMat;
using mindspore::dataset::LPixelType;
using mindspore::dataset::LDataType;
#define MS_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MSJNI", format, ##__VA_ARGS__)

bool ObjectBitmapToLiteMat(JNIEnv *env, const jobject &srcBitmap, LiteMat *lite_mat) {
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

bool ObjectPreProcessImageData(const LiteMat &lite_mat_bgr, LiteMat *lite_norm_mat_ptr) {
    bool ret = false;
    LiteMat lite_mat_resize;
    LiteMat &lite_norm_mat_cut = *lite_norm_mat_ptr;
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

    std::vector<float> means = {0.485, 0.456, 0.406};
    std::vector<float> stds = {0.229, 0.224, 0.225};
    SubStractMeanNormalize(lite_mat_convert_float, lite_norm_mat_cut, means, stds);
    return true;
}

char *ObjectCreateLocalModelBuffer(JNIEnv *env, jobject modelBuffer) {
    jbyte *modelAddr = static_cast<jbyte *>(env->GetDirectBufferAddress(modelBuffer));
    int modelLen = static_cast<int>(env->GetDirectBufferCapacity(modelBuffer));
    char *buffer(new char[modelLen]);
    memcpy(buffer, modelAddr, modelLen);
    return buffer;
}

/**
 *
 * @param msOutputs Model output,  the mindspore inferencing result.
 * @param srcImageWidth The width of the original input image.
 * @param srcImageHeight The height of the original input image.
 * @return
 */
std::string
ProcessRunnetResult(std::unordered_map<std::string, mindspore::tensor::MSTensor *> msOutputs,
                    int srcImageWidth, int srcImageHeight) {
    std::unordered_map<std::string, mindspore::tensor::MSTensor *>::iterator iter;
    iter = msOutputs.begin();
    auto branch2_string = iter->first;
    auto branch2_tensor = iter->second;

    ++iter;
    auto branch1_string = iter->first;
    auto branch1_tensor = iter->second;
    MS_PRINT("%s %s", branch1_string.c_str(), branch2_string.c_str());

    float *tmpscores2 = reinterpret_cast<float *>(branch1_tensor->MutableData());
    float *tmpdata = reinterpret_cast<float *>(branch2_tensor->MutableData());

    // Using ssd model util to process model branch outputs.
    SSDModelUtil ssdUtil(srcImageWidth, srcImageHeight);

    std::string retStr = ssdUtil.getDecodeResult(tmpscores2, tmpdata);
    MS_PRINT("retStr %s", retStr.c_str());

    return retStr;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_mindspore_imageobject_objectdetection_help_ObjectTrackingMobile_loadModel(JNIEnv *env,
                                                                                   jobject thiz,
                                                                                   jobject assetManager,
                                                                                   jobject buffer,
                                                                                   jint numThread) {
    MS_PRINT("MindSpore so version 20200730");
    if (nullptr == buffer) {
        MS_PRINT("error, buffer is nullptr!");
        return (jlong) nullptr;
    }
    jlong bufferLen = env->GetDirectBufferCapacity(buffer);
    MS_PRINT("MindSpore get bufferLen:%d", static_cast<int>(bufferLen));
    if (0 == bufferLen) {
        MS_PRINT("error, bufferLen is 0!");
        return (jlong) nullptr;
    }

    char *modelBuffer = ObjectCreateLocalModelBuffer(env, buffer);
    if (modelBuffer == nullptr) {
        MS_PRINT("modelBuffer create failed!");
        return (jlong) nullptr;
    }

    MS_PRINT("MindSpore loading Model.");
    void **labelEnv = new void *;
    MSNetWork *labelNet = new MSNetWork;
    *labelEnv = labelNet;

    mindspore::lite::Context *context = new mindspore::lite::Context;
    context->thread_num_ = numThread;

    labelNet->CreateSessionMS(modelBuffer, bufferLen, context);
    delete context;
    if (labelNet->session() == nullptr) {
        delete labelNet;
        delete labelEnv;
        MS_PRINT("MindSpore create session failed!.");
        return (jlong) nullptr;
    }
    MS_PRINT("MindSpore create session successfully.");

    if (buffer != nullptr) {
        env->DeleteLocalRef(buffer);
    }

    if (assetManager != nullptr) {
        env->DeleteLocalRef(assetManager);
    }
    MS_PRINT("ptr released successfully.");

    return (jlong) labelEnv;
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_mindspore_imageobject_objectdetection_help_ObjectTrackingMobile_runNet(JNIEnv *env,
                                                                                jobject thiz,
                                                                                jlong netEnv,
                                                                                jobject srcBitmap) {
    LiteMat lite_mat_bgr, lite_norm_mat_cut;

    if (!ObjectBitmapToLiteMat(env, srcBitmap, &lite_mat_bgr)) {
        MS_PRINT("ObjectBitmapToLiteMat error");
        return NULL;
    }
    int srcImageWidth = lite_mat_bgr.width_;
    int srcImageHeight = lite_mat_bgr.height_;
    if (!ObjectPreProcessImageData(lite_mat_bgr, &lite_norm_mat_cut)) {
        MS_PRINT("ObjectPreProcessImageData error");
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
    auto inTensor = msInputs.front();
    float *dataHWC = reinterpret_cast<float *>(lite_norm_mat_cut.data_ptr_);
    // copy input Tensor
    memcpy(inTensor->MutableData(), dataHWC,
           inputDims.channel * inputDims.width * inputDims.height * sizeof(float));
    MS_PRINT("MindSpore get msInputs.");

    auto status = mSession->RunGraph();
    if (status != mindspore::lite::RET_OK) {
        MS_PRINT("MindSpore runnet error.");
        return NULL;
    }

    auto names = mSession->GetOutputTensorNames();
    std::unordered_map<std::string,
            mindspore::tensor::MSTensor *> msOutputs;
    for (const auto &name : names) {
        auto temp_dat = mSession->GetOutputByTensorName(name);
        msOutputs.insert(std::pair<std::string, mindspore::tensor::MSTensor *>{name, temp_dat});
    }
    std::string retStr = ProcessRunnetResult(msOutputs, srcImageWidth, srcImageHeight);
    const char *resultChardata = retStr.c_str();

    return (env)->NewStringUTF(resultChardata);
}


extern "C"
JNIEXPORT jboolean JNICALL
Java_com_mindspore_imageobject_objectdetection_help_ObjectTrackingMobile_unloadModel(JNIEnv *env,
                                                                                     jobject thiz,
                                                                                     jlong netEnv) {
    void **labelEnv = reinterpret_cast<void **>(netEnv);
    MSNetWork *labelNet = static_cast<MSNetWork *>(*labelEnv);
    labelNet->ReleaseNets();
    return (jboolean) true;
}

