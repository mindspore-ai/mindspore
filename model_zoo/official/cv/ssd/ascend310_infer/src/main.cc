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
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/minddata/dataset/include/vision.h"
#include "include/minddata/dataset/include/execute.h"

#include "../inc/utils.h"

using mindspore::GlobalContext;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::ModelContext;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::dataset::Execute;
using mindspore::dataset::vision::DvppDecodeResizeCropJpeg;


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

    GlobalContext::SetGlobalDeviceTarget(mindspore::kDeviceTypeAscend310);
    GlobalContext::SetGlobalDeviceID(FLAGS_device_id);
    auto graph = Serialization::LoadModel(FLAGS_mindir_path, ModelType::kMindIR);
    auto model_context = std::make_shared<mindspore::ModelContext>();
    if (!FLAGS_aipp_path.empty()) {
      ModelContext::SetInsertOpConfigPath(model_context, FLAGS_aipp_path);
    }

    Model model(GraphCell(graph), model_context);
    Status ret = model.Build();
    if (ret != kSuccess) {
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
    Execute resize_op(DvppDecodeResizeCropJpeg({640, 640}, {640, 640}));
    for (size_t i = 0; i < size; ++i) {
        struct timeval start = {0};
        struct timeval end = {0};
        double startTimeMs;
        double endTimeMs;
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;
        std::cout << "Start predict input files:" << all_files[i] << std::endl;
        auto imgDvpp = std::make_shared<MSTensor>();
        resize_op(ReadFileToTensor(all_files[i]), imgDvpp.get());

        inputs.emplace_back(imgDvpp->Name(), imgDvpp->DataType(), imgDvpp->Shape(),
                            imgDvpp->Data().get(), imgDvpp->DataSize());
        gettimeofday(&start, nullptr);
        ret = model.Predict(inputs, &outputs);
        gettimeofday(&end, nullptr);
        if (ret != kSuccess) {
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
