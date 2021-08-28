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

#include "../inc/sample_process.h"
#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "../inc/model_process.h"
#include "acl/acl.h"
#include "../inc/utils.h"
extern bool g_is_device;
using std::string;
using std::vector;

SampleProcess::SampleProcess() :deviceId_(0), context_(nullptr), stream_(nullptr) {
}

SampleProcess::~SampleProcess() {
    DestroyResource();
}

Result SampleProcess::InitResource() {
    // ACL init

    const char *aclConfigPath = "./src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");

    // open device
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
    g_is_device = (runMode == ACL_DEVICE);
    INFO_LOG("get run mode success");
    return SUCCESS;
}

void SampleProcess::GetAllFiles(std::string path, std::vector<string> *files) {
    DIR *pDir = NULL;
    struct dirent* ptr = nullptr;
    if (!(pDir = opendir(path.c_str()))) {
        return;
    }
    while ((ptr = readdir(pDir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files->push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
}

Result SampleProcess::Process(char *om_path, char *input_folder) {
    // model init
    const double second_to_millisecond = 1000;
    const double second_to_microsecond = 1000000;

    double whole_cost_time = 0.0;
    struct timeval start_global = {0};
    struct timeval end_global = {0};
    double startTimeMs_global = 0.0;
    double endTimeMs_global = 0.0;

    gettimeofday(&start_global, nullptr);

    ModelProcess processModel;
    const char* omModelPath = om_path;

    Result ret = processModel.LoadModelFromFileWithMem(omModelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromFileWithMem failed");
        return FAILED;
    }

    ret = processModel.CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    ret = processModel.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }

    std::vector<string> testFile;
    GetAllFiles(input_folder, &testFile);

    if (testFile.size() == 0) {
        WARN_LOG("no input data under folder");
    }

    double model_cost_time = 0.0;
    double edge_to_edge_model_cost_time = 0.0;

    for (size_t index = 0; index < testFile.size(); ++index) {
        INFO_LOG("start to process file:%s", testFile[index].c_str());
        // model process

        struct timeval time_init = {0};
        double timeval_init = 0.0;
        gettimeofday(&time_init, nullptr);
        timeval_init = (time_init.tv_sec * second_to_microsecond + time_init.tv_usec) / second_to_millisecond;

        uint32_t devBufferSize;
        void *picDevBuffer = Utils::GetDeviceBufferOfFile(testFile[index], &devBufferSize);
        if (picDevBuffer == nullptr) {
            ERROR_LOG("get pic device buffer failed,index is %zu", index);
            return FAILED;
        }
        ret = processModel.CreateInput(picDevBuffer, devBufferSize);
        if (ret != SUCCESS) {
            ERROR_LOG("execute CreateInput failed");
            aclrtFree(picDevBuffer);
            return FAILED;
        }

        struct timeval start = {0};
        struct timeval end = {0};
        double startTimeMs = 0.0;
        double endTimeMs = 0.0;
        gettimeofday(&start, nullptr);
        startTimeMs = (start.tv_sec * second_to_microsecond + start.tv_usec) / second_to_millisecond;

        ret = processModel.Execute();

        gettimeofday(&end, nullptr);
        endTimeMs = (end.tv_sec * second_to_microsecond + end.tv_usec) / second_to_millisecond;

        double cost_time = endTimeMs - startTimeMs;
        INFO_LOG("model infer time: %lf ms", cost_time);

        model_cost_time += cost_time;

        double edge_to_edge_cost_time = endTimeMs - timeval_init;
        edge_to_edge_model_cost_time += edge_to_edge_cost_time;

        if (ret != SUCCESS) {
            ERROR_LOG("execute inference failed");
            aclrtFree(picDevBuffer);
            return FAILED;
        }

        int pos = testFile[index].find_last_of('/');
        std::string name = testFile[index].substr(pos+1);
        std::string outputname = name.substr(0, name.rfind("."));

        // dump output result to file in the current directory
        processModel.DumpModelOutputResult(const_cast<char *>(outputname.c_str()));

        // release model input buffer
        aclrtFree(picDevBuffer);
        processModel.DestroyInput();
    }
    double test_file_size = 0.0;
    test_file_size = testFile.size();
    INFO_LOG("infer dataset size:%lf", test_file_size);

    gettimeofday(&end_global, nullptr);
    startTimeMs_global = (start_global.tv_sec * second_to_microsecond + start_global.tv_usec) / second_to_millisecond;
    endTimeMs_global = (end_global.tv_sec * second_to_microsecond + end_global.tv_usec) / second_to_millisecond;
    whole_cost_time = (endTimeMs_global - startTimeMs_global) / test_file_size;

    model_cost_time /= test_file_size;
    INFO_LOG("model cost time per sample: %lf ms", model_cost_time);
    edge_to_edge_model_cost_time /= test_file_size;
    INFO_LOG("edge-to-edge model cost time per sample:%lf ms", edge_to_edge_model_cost_time);
    INFO_LOG("whole cost time per sample: %lf ms", whole_cost_time);

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

