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
#include "include/dataset/transforms.h"
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
using mindspore::dataset::vision::Decode;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::CenterCrop;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::HWC2CHW;


DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_name, "cifar10", "['cifar10', 'imagenet2012']");
DEFINE_string(input0_path, ".", "input0 path");
DEFINE_int32(device_id, 0, "device id");

int load_model(Model *model, std::vector<MSTensor> *model_inputs, std::string mindir_path, int device_id) {
  if (RealPath(mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  auto context = std::make_shared<Context>();
  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(device_id);
  context->MutableDeviceInfo().push_back(ascend310);
  mindspore::Graph graph;
  Serialization::Load(mindir_path, ModelType::kMindIR, &graph);

  Status ret = model->Build(GraphCell(graph), context);
  if (ret != kSuccess) {
    std::cout << "ERROR: Build failed." << std::endl;
    return 1;
  }

  *model_inputs = model->GetInputs();
  if (model_inputs->empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Model model;
  std::vector<MSTensor> model_inputs;
  load_model(&model, &model_inputs, FLAGS_mindir_path, FLAGS_device_id);

  std::map<double, double> costTime_map;
  struct timeval start = {0};
  struct timeval end = {0};
  double startTimeMs;
  double endTimeMs;

  if (FLAGS_dataset_name == "cifar10") {
    auto input0_files = GetAllFiles(FLAGS_input0_path);
    if (input0_files.empty()) {
      std::cout << "ERROR: no input data." << std::endl;
      return 1;
    }
    size_t size = input0_files.size();
    for (size_t i = 0; i < size; ++i) {
      std::vector<MSTensor> inputs;
      std::vector<MSTensor> outputs;
      std::cout << "Start predict input files:" << input0_files[i] <<std::endl;
      auto input0 = ReadFileToTensor(input0_files[i]);
      inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                          input0.Data().get(), input0.DataSize());

      gettimeofday(&start, nullptr);
      Status ret = model.Predict(inputs, &outputs);
      gettimeofday(&end, nullptr);
      if (ret != kSuccess) {
        std::cout << "Predict " << input0_files[i] << " failed." << std::endl;
        return 1;
      }
      startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
      endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
      costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
      WriteResult(input0_files[i], outputs);
    }
  } else {
    auto input0_files = GetAllInputData(FLAGS_input0_path);
    if (input0_files.empty()) {
      std::cout << "ERROR: no input data." << std::endl;
      return 1;
    }
    size_t size = input0_files.size();
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < input0_files[i].size(); ++j) {
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;
        std::cout << "Start predict input files:" << input0_files[i][j] <<std::endl;
        auto decode = Decode();
        auto resize = Resize({256, 256});
        auto centercrop = CenterCrop({224, 224});
        auto normalize = Normalize({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
        auto hwc2chw = HWC2CHW();

        Execute SingleOp({decode, resize, centercrop, normalize, hwc2chw});
        auto imgDvpp = std::make_shared<MSTensor>();
        SingleOp(ReadFileToTensor(input0_files[i][j]), imgDvpp.get());
        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                            imgDvpp->Data().get(), imgDvpp->DataSize());
      gettimeofday(&start, nullptr);
      Status ret = model.Predict(inputs, &outputs);
      gettimeofday(&end, nullptr);
      if (ret != kSuccess) {
        std::cout << "Predict " << input0_files[i][j] << " failed." << std::endl;
        return 1;
      }
      startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
      endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
      costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
      WriteResult(input0_files[i][j], outputs);
      }
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
