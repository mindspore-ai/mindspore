/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "AclProcess.h"
#include <sys/time.h>
#include <thread>
#include <string>

/*
 * @description Implementation of constructor for class AclProcess with parameter list
 * @attention context is passed in as a parameter after being created in ResourceManager::InitResource
 */
AclProcess::AclProcess(int deviceId, const std::string &om_path, uint32_t width, uint32_t height)
    : deviceId_(deviceId), stream_(nullptr), modelProcess_(nullptr), dvppCommon_(nullptr), keepRatio_(true) {
    modelInfo_.modelPath = om_path;
    modelInfo_.modelWidth = width;
    modelInfo_.modelHeight = height;
}

/*
 * @description Release all the resource
 * @attention context will be released in ResourceManager::Release
 */
void AclProcess::Release() {
    // Synchronize stream and release Dvpp channel
    dvppCommon_->DeInit();
    // Release stream
    if (stream_ != nullptr) {
        int ret = aclrtDestroyStream(stream_);
        if (ret != OK) {
            std::cout << "Failed to destroy the stream, ret = " << ret << ".";
        }
        stream_ = nullptr;
    }
    // Destroy resources of modelProcess_
    modelProcess_->DeInit();

    // Release Dvpp buffer
    dvppCommon_->ReleaseDvppBuffer();

    return;
}

/*
 * @description Initialize the modules used by this sample
 * @return int int code
 */
int AclProcess::InitModule() {
    // Create Dvpp common object
    if (dvppCommon_ == nullptr) {
        dvppCommon_ = std::make_shared<DvppCommon>(stream_);
        int retDvppCommon = dvppCommon_->Init();
        if (retDvppCommon != OK) {
            std::cout << "Failed to initialize dvppCommon, ret = " << retDvppCommon << std::endl;
            return retDvppCommon;
        }
    }
    // Create model inference object
    if (modelProcess_ == nullptr) {
        modelProcess_ = std::make_shared<ModelProcess>(deviceId_);
    }
    // Initialize ModelProcess module
    int ret = modelProcess_->Init(modelInfo_.modelPath);
    if (ret != OK) {
        std::cout << "Failed to initialize the model process module, ret = " << ret << "." << std::endl;
        return ret;
    }
    std::cout << "Initialized the model process module successfully." << std::endl;
    return OK;
}

/*
 * @description Create resource for this sample
 * @return int int code
 */
int AclProcess::InitResource() {
    int ret = aclInit(nullptr);  // Initialize ACL
    if (ret != OK) {
        std::cout << "Failed to init acl, ret = " << ret << std::endl;
        return ret;
    }

    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        std::cout << "acl set device " << deviceId_ << "intCode = "<< static_cast<int32_t>(ret) << std::endl;
        return ret;
    }
    std::cout << "set device "<< deviceId_ << " success" << std::endl;

    // create context (set current)
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_SUCCESS) {
        std::cout << "acl create context failed, deviceId = " << deviceId_ <<
            "intCode = "<< static_cast<int32_t>(ret) << std::endl;
        return ret;
    }
    std::cout << "create context success" << std::endl;

    ret = aclrtCreateStream(&stream_);  // Create stream for application
    if (ret != OK) {
        std::cout << "Failed to create the acl stream, ret = " << ret << "." << std::endl;
        return ret;
    }
    std::cout << "Created the acl stream successfully." << std::endl;
    // Initialize dvpp module
    if (InitModule() != OK) {
        return INIT_FAIL;
    }

    aclmdlDesc *modelDesc = modelProcess_->GetModelDesc();
    size_t outputSize = aclmdlGetNumOutputs(modelDesc);
    modelInfo_.outputNum = outputSize;
    for (size_t i = 0; i < outputSize; i++) {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
        void *outputBuffer = nullptr;
        ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != OK) {
            std::cout << "Failed to malloc buffer, ret = " << ret << "." << std::endl;
            return ret;
        }
        outputBuffers_.push_back(outputBuffer);
        outputSizes_.push_back(bufferSize);
    }

    return OK;
}

int AclProcess::WriteResult(const std::string& imageFile) {
    std::string homePath = "./result_Files";
    void *resHostBuf = nullptr;
    for (size_t i = 0; i < outputBuffers_.size(); ++i) {
        size_t output_size;
        void *netOutput;
        netOutput = outputBuffers_[i];
        output_size =  outputSizes_[i];
        int ret = aclrtMallocHost(&resHostBuf, output_size);
        if (ret != OK) {
            std::cout << "Failed to print the result, malloc host failed, ret = " << ret << "." << std::endl;
            return ret;
        }

        ret = aclrtMemcpy(resHostBuf, output_size, netOutput,
                          output_size, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != OK) {
            std::cout << "Failed to print result, memcpy device to host failed, ret = " << ret << "." << std::endl;
            return ret;
        }

        int pos = imageFile.rfind('/');
        std::string fileName(imageFile, pos + 1);
        fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'), "_" + std::to_string(i) + ".bin");

        std::string outFileName = homePath + "/" + fileName;
        try {
            FILE *outputFile = fopen(outFileName.c_str(), "wb");
            if (outputFile == nullptr) {
                std::cout << "open result file " << outFileName << " failed" << std::endl;
                return INVALID_POINTER;
            }
            size_t size = fwrite(resHostBuf, sizeof(char), output_size, outputFile);
            if (size != output_size) {
                fclose(outputFile);
                outputFile = nullptr;
                std::cout << "write result file " << outFileName << " failed, write size[" << size <<
                    "] is smaller than output size[" << output_size << "], maybe the disk is full." << std::endl;
                return ERROR;
            }

            fclose(outputFile);
            outputFile = nullptr;
        } catch (std::exception &e) {
            std::cout << "write result file " << outFileName << " failed, error info: " << e.what() << std::endl;
            std::exit(1);
        }

        ret = aclrtFreeHost(resHostBuf);
        if (ret != OK) {
            std::cout << "aclrtFree host output memory failed" << std::endl;
            return ret;
        }
    }
    return OK;
}

/**
 * Read a file, store it into the RawData structure
 *
 * @param filePath file to read to
 * @param fileData RawData structure to store in
 * @return OK if create success, int code otherwise
 */
int AclProcess::ReadFile(const std::string &filePath, RawData *fileData) {
    // Open file with reading mode
    FILE *fp = fopen(filePath.c_str(), "rb");
    if (fp == nullptr) {
        std::cout << "Failed to open file, filePath = " << filePath << std::endl;
        return OPEN_FILE_FAIL;
    }
    // Get the length of input file
    fseek(fp, 0, SEEK_END);
    size_t fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    // If file not empty, read it into FileInfo and return it
    if (fileSize > 0) {
        fileData->lenOfByte = fileSize;
        fileData->data = std::make_shared<uint8_t>();
        fileData->data.reset(new uint8_t[fileSize], std::default_delete<uint8_t[]>());
        uint32_t readRet = fread(fileData->data.get(), 1, fileSize, fp);
        if (readRet == 0) {
            fclose(fp);
            return READ_FILE_FAIL;
        }
        fclose(fp);
        return OK;
    }
    fclose(fp);
    return INVALID_PARAM;
}
/*
 * @description Preprocess the input image
 * @param imageFile input image path
 * @return int int code
 */
int AclProcess::Preprocess(const std::string& imageFile) {
    RawData imageInfo;
    int ret = ReadFile(imageFile, &imageInfo);  // Read image data from input image file
    if (ret != OK) {
        std::cout << "Failed to read file, ret = " << ret << "." << std::endl;
        return ret;
    }
    // Run process of jpegD
    ret = dvppCommon_->CombineJpegdProcess(imageInfo, PIXEL_FORMAT_YUV_SEMIPLANAR_420, true);
    if (ret !=  OK) {
        std::cout  << "Failed to execute image decoded of preprocess module, ret = " << ret << "." << std::endl;
        return ret;
    }
    // Get output of decode jpeg image
    std::shared_ptr<DvppDataInfo> decodeOutData = dvppCommon_->GetDecodedImage();
    // Run resize application function
    DvppDataInfo resizeOutData;
    resizeOutData.height = modelInfo_.modelHeight;
    resizeOutData.width = modelInfo_.modelWidth;
    resizeOutData.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    ret = dvppCommon_->CombineResizeProcess(decodeOutData, resizeOutData, true, VPC_PT_PADDING);
    if (ret != OK) {
        std::cout << "Failed to execute image resized of preprocess module, ret = " << ret << "." << std::endl;
        return ret;
    }

    RELEASE_DVPP_DATA(decodeOutData->data);
    return OK;
}

/*
 * @description Inference of model
 * @return int int code
 */
int AclProcess::ModelInfer(std::map<double, double> *costTime_map) {
    // Get output of resize module
    std::shared_ptr<DvppDataInfo> resizeOutData = dvppCommon_->GetResizedImage();
    std::shared_ptr<DvppDataInfo> inputImg = dvppCommon_->GetInputImage();

    float widthScale, heightScale;
    if (keepRatio_) {
        widthScale = static_cast<float>(resizeOutData->width) / inputImg->width;
        if (widthScale > static_cast<float>(resizeOutData->height) / inputImg->height) {
            widthScale = static_cast<float>(resizeOutData->height) / inputImg->height;
        }
        heightScale = widthScale;
    } else {
        widthScale = static_cast<float>(resizeOutData->width) / inputImg->width;
        heightScale = static_cast<float>(resizeOutData->height) / inputImg->height;
    }

    float im_info[4];
    im_info[0] = static_cast<float>(inputImg->height);
    im_info[1] = static_cast<float>(inputImg->width);
    im_info[2] = heightScale;
    im_info[3] = widthScale;
    void *imInfo_dst = nullptr;
    int ret = aclrtMalloc(&imInfo_dst, 16, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        std::cout << "aclrtMalloc failed, ret = " << ret << std::endl;
        aclrtFree(imInfo_dst);
        return ret;
    }
    ret = aclrtMemcpy(reinterpret_cast<uint8_t *>(imInfo_dst), 16, im_info, 16, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        std::cout << "aclrtMemcpy failed, ret = " << ret << std::endl;
        aclrtFree(imInfo_dst);
        return ret;
    }

    std::vector<void *> inputBuffers({resizeOutData->data, imInfo_dst});
    std::vector<size_t> inputSizes({resizeOutData->dataSize, 4 * 4});

    for (size_t i = 0; i < modelInfo_.outputNum; i++) {
        aclrtMemset(outputBuffers_[i], outputSizes_[i], 0, outputSizes_[i]);
    }
    // Execute classification model
    ret = modelProcess_->ModelInference(inputBuffers, inputSizes, outputBuffers_, outputSizes_, costTime_map);
    if (ret != OK) {
        aclrtFree(imInfo_dst);
        std::cout << "Failed to execute the classification model, ret = " << ret << "." << std::endl;
        return ret;
    }

    ret = aclrtFree(imInfo_dst);
    if (ret != OK) {
        std::cout << "aclrtFree image info failed" << std::endl;
        return ret;
    }
    RELEASE_DVPP_DATA(resizeOutData->data);
    return OK;
}

/*
 * @description Process classification
 *
 * @par Function
 * 1.Dvpp module preprocess
 * 2.Execute classification model
 * 3.Execute single operator
 * 4.Write result
 *
 * @param imageFile input file path
 * @return int int code
 */

int AclProcess::Process(const std::string& imageFile, std::map<double, double> *costTime_map) {
    struct timeval begin = {0};
    struct timeval end = {0};
    gettimeofday(&begin, nullptr);

    int ret = Preprocess(imageFile);
    if (ret != OK) {
        return ret;
    }

    ret = ModelInfer(costTime_map);
    if (ret != OK) {
        return ret;
    }

    ret = WriteResult(imageFile);
    if (ret != OK) {
        std::cout << "write result failed." << std::endl;
        return ret;
    }
    gettimeofday(&end, nullptr);

    const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
    std::cout << "[Process Delay] cost: " << costMs << "ms." << std::endl;
    return OK;
}
