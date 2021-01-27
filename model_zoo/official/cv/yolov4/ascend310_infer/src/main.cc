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
#include "include/api/serialization.h"
#include "include/api/context.h"
#include "minddata/dataset/include/minddata_eager.h"
#include "../inc/utils.h"
#include "include/api/types.h"
#include "minddata/dataset/include/vision.h"

using mindspore::api::Context;
using mindspore::api::Serialization;
using mindspore::api::Model;
using mindspore::api::kModelOptionInsertOpCfgPath;
using mindspore::api::kModelOptionPrecisionMode;
using mindspore::api::kModelOptionOpSelectImplMode;
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
DEFINE_string(precision_mode, "allow_fp32_to_fp16", "precision mode");
DEFINE_string(op_select_impl_mode, "", "op select impl mode");
DEFINE_string(input_shape, "img_data:1, 3, 768, 1280; img_info:1, 4", "input shape");
DEFINE_string(input_format, "nchw", "input format");
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
    if (!FLAGS_precision_mode.empty()) {
        build_options.emplace(kModelOptionPrecisionMode, FLAGS_precision_mode);
    }
    if (!FLAGS_op_select_impl_mode.empty()) {
        build_options.emplace(kModelOptionOpSelectImplMode, FLAGS_op_select_impl_mode);
    }

    if (!FLAGS_aipp_path.empty()) {
        build_options.emplace(kModelOptionInsertOpCfgPath, FLAGS_aipp_path);
    }

    Status ret = model.Build(build_options);
    if (ret != SUCCESS) {
        std::cout << "EEEEEEEERROR Build failed." << std::endl;
        return 1;
    }

    auto all_files = GetAllFiles(FLAGS_dataset_path);
    if (all_files.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return 1;
    }

    std::map<double, double> costTime_map;
    size_t size = all_files.size();
    MindDataEager SingleOp({DvppDecodeResizeJpeg({608, 608})});
    for (size_t i = 0; i < size; ++i) {
        struct timeval start = {0};
        struct timeval end = {0};
        double startTime_ms;
        double endTime_ms;
        std::vector<Buffer> inputs;
        std::vector<Buffer> outputs;
        std::cout << "Start predict input files:" << all_files[i] << std::endl;
        auto imgDvpp = SingleOp(ReadFileToTensor(all_files[i]));
        std::vector<float> input_shape = {608, 608};

        inputs.clear();
        inputs.emplace_back(imgDvpp->Data(), imgDvpp->DataSize());
        inputs.emplace_back(input_shape.data(), input_shape.size() * sizeof(float));
        gettimeofday(&start, NULL);
        ret = model.Predict(inputs, &outputs);
        gettimeofday(&end, NULL);
        if (ret != SUCCESS) {
            std::cout << "Predict " << all_files[i] << " failed." << std::endl;
            return 1;
        }
        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
        WriteResult(all_files[i], outputs);
    }
    double average = 0.0;
    int infer_cnt = 0;
    char tmpCh[256] = {0};
    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        average += diff;
        infer_cnt++;
    }
    average = average/infer_cnt;
    snprintf(tmpCh, sizeof(tmpCh), "NN inference cost average time: %4.3f ms of infer_count %d \n", average, infer_cnt);
    std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << infer_cnt << std::endl;
    std::string file_name = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
    file_stream << tmpCh;
    file_stream.close();
    costTime_map.clear();
    return 0;
}
