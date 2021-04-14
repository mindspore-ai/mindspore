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

#include <dirent.h>
#include <sys/time.h>
#include <time.h>
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <unordered_map>
#include <iterator>
#include <thread>
#include <sstream>

#include "acl/acl.h"
#include "../inc/utils.h"
#include "../inc/model_process.h"
#include "../inc/sample_process.h"

extern bool g_isDevice;

SampleProcess::SampleProcess() :deviceId_(0), context_(nullptr), stream_(nullptr), threadNum_(0) {}

SampleProcess::SampleProcess(uint32_t deviceId, uint32_t threadNum):
    deviceId_(deviceId),
    threadNum_(threadNum),
    context_(nullptr),
    stream_(nullptr) {}

SampleProcess::~SampleProcess() {
    DestroyResource();
}

Result SampleProcess::InitResource() {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");

    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl open device %d failed", deviceId_);
        return FAILED;
    }
    INFO_LOG("open device %d success", deviceId_);

    // create context (set current)
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create context failed");
        return FAILED;
    }
    INFO_LOG("create context success");

    // create stream
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");

    // get run mode
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("get run mode success");

    return SUCCESS;
}

Result SampleProcess::CreateModelProcessInstance(std::vector<std::string> omPaths,
                                                 std::vector<std::string> inputDataPaths,
                                                 std::vector<std::string> inputIdPaths,
                                                 uint32_t batchSize) {
    for (int i = 0; i < omPaths.size(); ++i) {
        std::cout << "om_path : " << omPaths[i]  << std::endl;
        auto processModel = std::make_shared<ModelProcess>(inputDataPaths[i], inputIdPaths[i], batchSize);
        Result ret = processModel->LoadModelFromFileWithMem(omPaths[i].c_str());
        if (ret != SUCCESS) {
            ERROR_LOG("load model from file failed");
            return FAILED;
        }

        ret = processModel->InitResource();
        if (ret != SUCCESS) {
            ERROR_LOG("create model description failed");
            return FAILED;
        }

        modelProcessContainer_.emplace(i, processModel);
    }

    return SUCCESS;
}

std::vector<std::vector<std::vector<int>>> SampleProcess::ReadHistory(std::vector<std::string> historyFile,
                                                                      uint32_t batchSize) {
    std::vector<std::vector<std::vector<int>>> allHistory;
    for (auto &file : historyFile) {
        std::vector<std::vector<int>> history(batchSize, std::vector<int>(50));
        Utils::ReadFileToVector(file, batchSize, 50, &history);
        allHistory.emplace_back(history);
    }

    return allHistory;
}

Result SampleProcess::Process(const std::vector<std::string> &omPaths,
                              const std::vector<std::string> &inputDataPaths,
                              const std::vector<std::string> &inputIdPaths,
                              const std::string &browsedNewsPath,
                              uint32_t batchSize) {
    struct timeval totalStart;
    struct timeval totalEnd;

    CreateModelProcessInstance(omPaths, inputDataPaths, inputIdPaths, batchSize);

    uint32_t fileNum = modelProcessContainer_[0]->ReadFiles();
    std::string historyDir = modelProcessContainer_[1]->GetInputDataPath() + "/00_history_data";
    std::vector<std::string> historyFile = Utils::GetAllBins(historyDir);

    size_t historySize = historyFile.size();
    std::vector<std::vector<std::vector<int>>> allHistory = ReadHistory(historyFile, batchSize);

    uint32_t browsedFileNum = ReadBrowsedData(browsedNewsPath);

    gettimeofday(&totalStart, NULL);
    modelProcessContainer_[0]->ExecuteWithFile(fileNum);

    std::map<int, void *> result = modelProcessContainer_[0]->GetResult();

    std::vector<uint32_t> model1OutputBuffSize = modelProcessContainer_[0]->GetOutputSize();
    std::vector<uint32_t> inputBuffSize = modelProcessContainer_[1]->GetInputSize();

    uint32_t singleDatsSize = model1OutputBuffSize[0] / batchSize;
    void* browedNews = NULL;
    aclrtMallocHost(&browedNews, inputBuffSize[0]);

    struct timeval start;
    struct timeval end;
    double startTime_ms;
    double endTime_ms;
    for (int i = 0; i < historySize; ++i) {
        gettimeofday(&start, NULL);
        for (int j = 0; j < 16; ++j) {
            for (int k = 0; k < 50; ++k) {
                auto it = result.find(allHistory[i][j][k]);
                if (it != result.end()) {
                    aclrtMemcpy(reinterpret_cast<uint8_t *>(browedNews) + (j * 50 + k) * singleDatsSize, singleDatsSize,
                                result[allHistory[i][j][k]], singleDatsSize, ACL_MEMCPY_HOST_TO_HOST);
                }
            }
        }
        modelProcessContainer_[1]->CpyDataToDevice(browedNews, inputBuffSize[0], 0);
        modelProcessContainer_[1]->Execute(i);
        gettimeofday(&end, NULL);
        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        secondModelCostTime_map_.insert(std::pair<double, double>(startTime_ms, endTime_ms));
    }

    GetPred(browsedFileNum);
    gettimeofday(&totalEnd, NULL);
    startTime_ms = (1.0 * totalStart.tv_sec * 1000000 + totalStart.tv_usec) / 1000;
    endTime_ms = (1.0 * totalEnd.tv_sec * 1000000 + totalEnd.tv_usec) / 1000;
    totalCostTime_map_.insert(std::pair<double, double>(startTime_ms, endTime_ms));
    aclrtFreeHost(browedNews);
    return SUCCESS;
}

Result SampleProcess::ReadBrowsedFile(const std::string &browsedNewsPath,
                                      std::vector<std::string> userIdFiles,
                                      std::vector<std::vector<int>> *usersIds,
                                      std::vector<std::vector<int>> *candidateNewsIds) {
    std::vector<int> candidateNewsId;
    std::vector<int> usersId;
    for (auto file : userIdFiles) {
        candidateNewsId.clear();
        usersId.clear();
        std::size_t pos = file.rfind("/");
        std::string name = file.substr(pos);

        std::string newsIdFileName = browsedNewsPath + "/01_candidate_nid_data" + name;

        Utils::ReadFileToVector(file, &usersId);
        Utils::ReadFileToVector(newsIdFileName, &candidateNewsId);

        usersIds->emplace_back(usersId);
        candidateNewsIds->emplace_back(candidateNewsId);
    }
    return SUCCESS;
}

uint32_t SampleProcess::ReadBrowsedData(const std::string &browsedNewsPath) {
    userIdFiles_ = Utils::GetAllBins(browsedNewsPath + "/00_user_id_data");
    ReadBrowsedFile(browsedNewsPath, userIdFiles_, &usersIds_, &candidateNewsIds_);
    uint32_t fileNum = userIdFiles_.size();

    return fileNum;
}

Result SampleProcess::GetPred(uint32_t fileNum) {
    std::map<int, void *> newsEncodeResult = modelProcessContainer_[0]->GetResult();
    std::map<int, void *> userEncodeResult = modelProcessContainer_[1]->GetResult();

    uint32_t perThreadNum = fileNum / threadNum_;
    std::vector<std::thread> threads;

    for (int i = 0; i < threadNum_; ++i) {
        if (i != threadNum_ - 1) {
            threads.emplace_back(std::thread(&SampleProcess::GetResult, this,
                                             i * perThreadNum, (i + 1) * perThreadNum,
                                             newsEncodeResult,
                                             userEncodeResult));
        } else {
            threads.emplace_back(std::thread(&SampleProcess::GetResult, this,
                                             i * perThreadNum,
                                             fileNum,
                                             newsEncodeResult,
                                             userEncodeResult));
        }
    }
    for (int i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }

    return SUCCESS;
}
void SampleProcess::GetResult(uint32_t startPos, uint32_t endPos,
                              std::map<int, void *> newsEncodeResult,
                              std::map<int, void *> userEncodeResult) {
    for (int i = startPos; i < endPos; ++i) {
        std::vector<std::vector<float>> newsCandidate;
        std::vector<float> userEncodeIds(400);
        for (int j = 0; j < candidateNewsIds_[i].size(); ++j) {
            std::vector<float> newsResults(400);
            float *newsResult = reinterpret_cast<float *>(newsEncodeResult[candidateNewsIds_[i][j]]);
            std::copy(newsResult, newsResult + 400, newsResults.begin());
            newsCandidate.emplace_back(newsResults);
        }
        float *userResult = reinterpret_cast<float *>(userEncodeResult[usersIds_[i][0]]);
        std::copy(userResult, userResult + 400, userEncodeIds.begin());

        std::vector<float> predResult;
        for (int j = 0; j < newsCandidate.size(); ++j) {
            float dotMulResult = 0;
            for (int k = 0; k < 400; ++k) {
                dotMulResult += newsCandidate[j][k] * userEncodeIds[k];
            }
            predResult.emplace_back(dotMulResult);
        }
        mtx_.lock();
        WriteResult(userIdFiles_[i], predResult, predResult.size() * 4);
        mtx_.unlock();
    }

    return;
}

int SampleProcess::WriteResult(const std::string& imageFile, std::vector<float> result, uint32_t size) {
    std::string homePath = "./result_Files/";
    std::size_t pos = imageFile.rfind("/");
    std::string name = imageFile.substr(pos);
    for (size_t i = 0; i < 1; ++i) {
        std::string outFileName = homePath + "/" + name;
        try {
            FILE *outputFile = fopen(outFileName.c_str(), "wb");
            fwrite(static_cast<void *>(&result[0]), size, sizeof(char), outputFile);
            fclose(outputFile);
            outputFile = nullptr;
        } catch (std::exception &e) {
            std::cout << "write result file " << outFileName << " failed, error info: " << e.what() << std::endl;
            return FAILED;
        }
    }
    return SUCCESS;
}

void SampleProcess::DestroyResource() {
    aclError ret;
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device is %d", deviceId_);

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");
}

std::vector<std::string> SampleProcess::GetModelExecCostTimeInfo() {
    std::vector<std::string> result;

    result.emplace_back(modelProcessContainer_[0]->GetCostTimeInfo());
    double secondModelAverage = 0.0;
    int infer_cnt = 0;

    for (auto iter = secondModelCostTime_map_.begin(); iter != secondModelCostTime_map_.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        secondModelAverage += diff;
        infer_cnt++;
    }
    secondModelAverage = secondModelAverage / infer_cnt;
    std::stringstream timeCost;
    timeCost << "second model inference cost average time: "<< secondModelAverage <<
        " ms of infer_count " << infer_cnt << std::endl;
    result.emplace_back(timeCost.str());

    double totalCostTime;
    totalCostTime = totalCostTime_map_.begin()->second - totalCostTime_map_.begin()->first;
    std::stringstream totalTimeCost;
    totalTimeCost << "total inference cost time: "<< totalCostTime << " ms; count " << infer_cnt << std::endl;
    result.emplace_back(totalTimeCost.str());

    return result;
}
