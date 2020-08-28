/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2018-2019. All rights reserved.
 */
#include <android/bitmap.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>
#include <cstring>
#include <set>

#include <MindSpore/errorcode.h>
#include <MindSpore/ms_tensor.h>
#include "MindSporeNetnative.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "MindSpore/MSNetWork.h"
#include "HMS/HMS_label_thres.h"

using namespace cv;
using namespace mindspore;
using namespace mindspore::tensor;

#define MS_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MSJNI", format, ##__VA_ARGS__)


void BitmapToMat2(JNIEnv *env, jobject &bitmap, Mat &mat, jboolean needUnPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = nullptr;
    Mat &dst = mat;
    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
    CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
              info.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
    CV_Assert(pixels);

    dst.create(info.height, info.width, CV_8UC4);
    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        Mat tmp(info.height, info.width, CV_8UC4, pixels);
        if (needUnPremultiplyAlpha) {
            cvtColor(tmp, dst, COLOR_RGBA2BGR);
        } else {

            tmp.copyTo(dst);
        }
    } else {
        Mat tmp(info.height, info.width, CV_8UC4, pixels);
        cvtColor(tmp, dst, COLOR_BGR5652RGBA);
    }
    AndroidBitmap_unlockPixels(env, bitmap);
    return;
}

void BitmapToMat(JNIEnv *env, jobject &bitmap, Mat &mat) {
    BitmapToMat2(env, bitmap, mat, true);
}

/**
 * Processing image with resize and normalize.
 */
cv::Mat PreProcessImageData(cv::Mat input) {
    cv::Mat imgFloatTmp, imgResized256, imgResized224;
    int resizeWidth = 256;
    int resizeHeight = 256;
    float normalizMin = 1.0;
    float normalizMax = 255.0;

    cv::resize(input, imgFloatTmp, cv::Size(resizeWidth, resizeHeight));


    imgFloatTmp.convertTo(imgResized256, CV_32FC3, normalizMin / normalizMax);

    int offsetX = 16;
    int offsetY = 16;
    int cropWidth = 224;
    int cropHeight = 224;

    // Standardization processing.
    float meanR = 0.485;
    float meanG = 0.456;
    float meanB = 0.406;
    float varR = 0.229;
    float varG = 0.224;
    float varB = 0.225;

    cv::Rect roi;
    roi.x = offsetX;
    roi.y = offsetY;
    roi.width = cropWidth;
    roi.height = cropHeight;

    // The final image size of the incoming model is 224*224.
    imgResized256(roi).copyTo(imgResized224);

    Scalar mean = Scalar(meanR, meanG, meanB);
    Scalar var = Scalar(varR, varG, varB);
    cv::Mat imgResized1;
    cv::Mat imgResized2;
    Mat imgMean(imgResized224.size(), CV_32FC3,
                mean);  // imgMean Each pixel channel is (0.485, 0.456, 0.406)
    Mat imgVar(imgResized224.size(), CV_32FC3,
               var); // imgVar Each pixel channel is (0.229, 0.224, 0.225)
    imgResized1 = imgResized224 - imgMean;
    imgResized2 = imgResized1 / imgVar;
    return imgResized2;
}

char *CreateLocalModelBuffer(JNIEnv *env, jobject modelBuffer) {
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
std::string ProcessRunnetResult(
        std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> msOutputs) {

    // Get the branch of the model output.
    // Use iterators to get map elements.
    std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>>::iterator iter;
    iter = msOutputs.begin();

    // The mobilenetv2.ms model output just one branch.
    auto outputString = iter->first;
    auto outputTensor = iter->second;

    int tensorNum = outputTensor[0]->ElementsNum();
    MS_PRINT("Number of tensor elements:%d", tensorNum);

    // Get a pointer to the first score.
    float *temp_scores = static_cast<float * >(outputTensor[0]->MutableData());

    float scores[RET_CATEGORY_SUM];
    for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
        if (temp_scores[i] > 0.5) {
            MS_PRINT("MindSpore scores[%d] : [%f]", i, temp_scores[i]);
        }
        scores[i] = temp_scores[i];
    }

    // Score for each category.
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


/**
 * The Java layer reads the model into MappedByteBuffer or ByteBuffer to load the model.
 */
extern "C"
JNIEXPORT jlong JNICALL
Java_com_huawei_himindsporedemo_gallery_classify_TrackingMobile_loadModel(JNIEnv *env, jobject thiz,
                                                                          jobject model_buffer,
                                                                          jint num_thread) {
    // TODO: implement loadModel()
    if (nullptr == model_buffer) {
        MS_PRINT("error, buffer is nullptr!");
        return (jlong) nullptr;
    }
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

    // To create a mindspore network inference environment.
    void **labelEnv = new void *;
    MSNetWork *labelNet = new MSNetWork;
    *labelEnv = labelNet;

    lite::Context *context = new lite::Context;
    context->thread_num_ = num_thread;

    labelNet->CreateSessionMS(modelBuffer, bufferLen, context);
    delete (context);

    if (labelNet->session == nullptr) {
        MS_PRINT("MindSpore create session failed!.");
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
Java_com_huawei_himindsporedemo_gallery_classify_TrackingMobile_runNet(JNIEnv *env, jclass type,
                                                                       jlong netEnv,
                                                                       jobject srcBitmap) {

    Mat matImageSrc;
    BitmapToMat(env, srcBitmap, matImageSrc);
    Mat matImgPreprocessed = PreProcessImageData(matImageSrc);


    ImgDims inputDims;
    inputDims.channel = matImgPreprocessed.channels();
    inputDims.width = matImgPreprocessed.cols;
    inputDims.height = matImgPreprocessed.rows;

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
    if (msInputs.size() == 0) {
        MS_PRINT("MindSpore error, msInputs.size() equals 0.");
        return NULL;
    }
    auto inTensor = msInputs.front();

    // dataHWC is the tensor format.
    float *dataHWC = new float[inputDims.channel * inputDims.width * inputDims.height];
    float *ptrTmp = reinterpret_cast<float *>(matImgPreprocessed.data);
    for (int i = 0; i < inputDims.channel * inputDims.width * inputDims.height; ++i) {
        dataHWC[i] = ptrTmp[i];
    }

    // Copy dataHWC to the model input tensor.
    memcpy(inTensor->MutableData(), dataHWC,
           inputDims.channel * inputDims.width * inputDims.height * sizeof(float));
    // When using 'new' to allocate memory space, we need to use 'delete' to free space.
    delete[] (dataHWC);

    // After the model and image tensor data is loaded, run inference.
    auto status = mSession->RunGraph();

    if (status != lite::RET_OK) {
        MS_PRINT("MindSpore run net error.");
        return NULL;
    }

    /**
     * Get the mindspore inference results.
     * Return the map of output node name and MindSpore Lite MSTensor.
     */
    auto msOutputs = mSession->GetOutputMapByNode();

    std::string resultStr = ProcessRunnetResult(msOutputs);

    const char *resultCharData = resultStr.c_str();
    return (env)->NewStringUTF(resultCharData);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_huawei_himindsporedemo_gallery_classify_TrackingMobile_unloadModel(JNIEnv *env,
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
