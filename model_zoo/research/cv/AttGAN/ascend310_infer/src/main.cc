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
#include <sstream>

#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/dataset/vision_ascend.h"
#include "include/dataset/execute.h"
#include "include/dataset/vision.h"
#include "inc/utils.h"

using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::dataset::Execute;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::HWC2CHW;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::Decode;
using color_rep_type = std::underlying_type<mindspore::DataType>::type;

DEFINE_string(gen_mindir_path, "", "generator mindir path");
DEFINE_string(dataset_path, "", "dataset path");
DEFINE_string(attr_file_path, "", "attribute file path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_int32(image_height, 128, "image height");
DEFINE_int32(image_width, 128, "image width");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (RealPath(FLAGS_gen_mindir_path).empty()) {
        std::cout << "Invalid generator mindir" << std::endl;
        return 1;
    }
    auto context = std::make_shared<Context>();
    auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
    ascend310->SetDeviceID(FLAGS_device_id);
    ascend310->SetBufferOptimizeMode("off_optimize");
    context->MutableDeviceInfo().push_back(ascend310);

    mindspore::Graph gen_graph;
    Serialization::Load(FLAGS_gen_mindir_path, ModelType::kMindIR, &gen_graph);

    Model gen_model;
    Status gen_ret = gen_model.Build(GraphCell(gen_graph), context);

    if (gen_ret != kSuccess) {
        std::cout << "ERROR: Generator build failed." << std::endl;
        return 1;
    }

    size_t n_attrs = 13;
    auto all_cfg = ReadCfgToTensor(FLAGS_attr_file_path, &n_attrs);
    auto all_files = GetAllFiles(FLAGS_dataset_path);
    std::map<double, double> costTime_map;
    double startTimeMs;
    double endTimeMs;
    size_t size = all_files.size();

    for (size_t i = 0; i < size; ++i) {
        struct timeval start = {0};
        struct timeval end = {0};
        std::cout << "Start predict input files:" << all_files[i] << std::endl;

        auto img = std::make_shared<MSTensor>();
        std::shared_ptr<TensorTransform> decode(new Decode());
        std::shared_ptr<TensorTransform> hwc2chw(new HWC2CHW());
        auto resizeShape = {FLAGS_image_height, FLAGS_image_width};
        std::shared_ptr<TensorTransform> resize(new Resize(resizeShape));
        std::shared_ptr<TensorTransform> normalize(
        new Normalize({127.5, 127.5, 127.5}, {127.5, 127.5, 127.5}));
        Execute composeDecode({decode, resize, normalize, hwc2chw});
        auto image = ReadFileToTensor(all_files[i]);
        composeDecode(image, img.get());

        for (size_t k = 0; k < n_attrs; ++k) {
            gettimeofday(&start, nullptr);
            std::vector<MSTensor> inputs;
            std::vector<MSTensor> outputs;
            size_t index = i * n_attrs + k;
            std::cout << static_cast<color_rep_type>(img->DataType()) << std::endl;
            inputs.emplace_back(img->Name(), img->DataType(), img->Shape(), img->Data().get(), img->DataSize());
            inputs.emplace_back(all_cfg[index].Name(), all_cfg[index].DataType(), all_cfg[index].Shape(),
                                all_cfg[index].Data().get(), all_cfg[index].DataSize());
            Status gen_model_ret = gen_model.Predict(inputs, &outputs);
            if (gen_model_ret != kSuccess) {
                std::cout << "Generator inference " << all_files[i] << " failed." << std::endl;
                return 1;
            }
            Denorm(&outputs);
            int pos = all_files[i].find('.');
            std::string fileName = all_files[i].substr(0, pos);
            WriteResult(fileName + "_" + std::to_string(k) + ".jpg", outputs);
            gettimeofday(&end, nullptr);
            startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
            endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
            costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
        }
    }
    double average = 0.0;
    int inferCount = 0;

    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        average += diff;
        inferCount++;
    }
    average = average / inferCount;
    std::stringstream timeCost;
    timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << inferCount << std::endl;
    std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << inferCount << std::endl;
    std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
    fileStream << timeCost.str();
    fileStream.close();
    costTime_map.clear();
    return 0;
}
