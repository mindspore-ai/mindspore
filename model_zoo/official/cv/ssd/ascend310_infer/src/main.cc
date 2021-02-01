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
#include <sys/time.h>
#include <gflags/gflags.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>

#include "include/api/model.h"
#include "include/api/context.h"
#include "minddata/dataset/include/minddata_eager.h"
#include "../inc/utils.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "minddata/dataset/include/vision.h"

using mindspore::api::Context;
using mindspore::api::Serialization;
using mindspore::api::Model;
using mindspore::api::kModelOptionInsertOpCfgPath;
using mindspore::api::Status;
using mindspore::api::MindDataEager;
using mindspore::api::Buffer;
using mindspore::api::ModelType;
using mindspore::api::GraphCell;
using mindspore::api::SUCCESS;
using mindspore::dataset::vision::DvppDecodeResizeJpeg;

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(aipp_path, "./aipp.cfg", "aipp path");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (RealPath(FLAGS_mindir_path).empty()) {
        std::cout << "Invalid mindir" << std::endl;
        return 1;
    }
    if (RealPath(FLAGS_aipp_path).empty()) {
        std::cout << "Invalid aipp path" << std::endl;
        return 1;
    }

    Context::Instance().SetDeviceTarget("Ascend310").SetDeviceID(FLAGS_device_id);
    auto graph = Serialization::LoadModel(FLAGS_mindir_path, ModelType::kMindIR);
    Model model((GraphCell(graph)));
    std::map<std::string, std::string> build_options;
    if (!FLAGS_aipp_path.empty()) {
        build_options.emplace(kModelOptionInsertOpCfgPath, FLAGS_aipp_path);
    }

    Status ret = model.Build(build_options);
    if (ret != SUCCESS) {
        std::cout << "ERROR: Build failed." << std::endl;
        return 1;
    }

    auto all_files = GetAllFiles(FLAGS_dataset_path);
    if (all_files.empty()) {
    std::cout << "ERROR: no input data." << std::endl;
    return 1;
    }

    std::map<double, double> costTime_map;
    size_t size = all_files.size();
    MindDataEager SingleOp({DvppDecodeResizeJpeg({640, 640})});
    for (size_t i = 0; i < size; ++i) {
        struct timeval start = {0};
        struct timeval end = {0};
        double startTimeMs;
        double endTimeMs;
        std::vector<Buffer> inputs;
        std::vector<Buffer> outputs;
        std::cout << "Start predict input files:" << all_files[i] << std::endl;
        auto imgDvpp = SingleOp(ReadFileToTensor(all_files[i]));

        inputs.emplace_back(imgDvpp->Data(), imgDvpp->DataSize());
        gettimeofday(&start, nullptr);
        ret = model.Predict(inputs, &outputs);
        gettimeofday(&end, nullptr);
        if (ret != SUCCESS) {
            std::cout << "Predict " << all_files[i] << " failed." << std::endl;
            return 1;
        }
        startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
        WriteResult(all_files[i], outputs);
    }
    double average = 0.0;
    int inferCount = 0;
    char tmpCh[256] = {0};
    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        average += diff;
        inferCount++;
    }
    average = average / inferCount;
    snprintf(tmpCh, sizeof(tmpCh), \
    "NN inference cost average time: %4.3f ms of infer_count %d \n", average, inferCount);
    std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << inferCount << std::endl;
    std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
    fileStream << tmpCh;
    fileStream.close();
    costTime_map.clear();
    return 0;
}
