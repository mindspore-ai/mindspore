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

#include <gflags/gflags.h>
#include <string>
#include <iostream>
#include <memory>
#include <fstream>
#include "../inc/utils.h"
#include "../inc/sample_process.h"

bool g_isDevice = false;

DEFINE_string(news_om_path, "../model/relu_double_geir.om", "om model path.");
DEFINE_string(user_om_path, "../model/relu_double_geir.om", "om model path.");
DEFINE_string(news_dataset_path, "../data", "input data dir");
DEFINE_string(user_dataset_path, "../data", "input data dir");
DEFINE_string(newsid_data_path, "../data", "input data dir");
DEFINE_string(userid_data_path, "../data", "input data dir");
DEFINE_string(browsed_news_path, "../data", "input data dir");
DEFINE_int32(batch_size, 16, "batch size");
DEFINE_int32(device_id, 0, "device id");
DEFINE_int32(thread_num, 8, "thread num");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << "news OM File Path :" << FLAGS_news_om_path << std::endl;
    std::cout << "user OM File Path :" << FLAGS_user_om_path << std::endl;
    std::cout << "news Dataset Path :" << FLAGS_news_dataset_path << std::endl;
    std::cout << "user Dataset Path :" << FLAGS_user_dataset_path << std::endl;
    std::cout << "browsed_news_path Path :" << FLAGS_browsed_news_path << std::endl;
    std::cout << "batch size :" << FLAGS_batch_size << std::endl;
    std::cout << "device id :" << FLAGS_device_id << std::endl;
    std::cout << "thread num :" << FLAGS_thread_num << std::endl;

    std::vector<std::string> omPaths;
    std::vector<std::string> datasetPaths;
    std::vector<std::string> idsPaths;
    omPaths.emplace_back(FLAGS_news_om_path);
    omPaths.emplace_back(FLAGS_user_om_path);
    datasetPaths.emplace_back(FLAGS_news_dataset_path);
    datasetPaths.emplace_back(FLAGS_user_dataset_path);
    idsPaths.emplace_back(FLAGS_newsid_data_path);
    idsPaths.emplace_back(FLAGS_userid_data_path);

    SampleProcess processSample(FLAGS_device_id, FLAGS_thread_num);
    Result ret = processSample.InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("sample init resource failed");
        return FAILED;
    }

    ret = processSample.Process(omPaths, datasetPaths, idsPaths, FLAGS_browsed_news_path, FLAGS_batch_size);
    if (ret != SUCCESS) {
        ERROR_LOG("sample process failed");
        return FAILED;
    }

    std::vector<std::string> costTime = processSample.GetModelExecCostTimeInfo();
    std::string file_name = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
    for (auto cost : costTime) {
        std::cout << cost << std::endl;
        file_stream << cost << std::endl;
    }

    file_stream.close();

    INFO_LOG("execute sample success");
    return SUCCESS;
}
