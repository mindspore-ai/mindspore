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


DEFINE_string(simclr_classifier_mindir_path, "", "simclr_classifier mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (RealPath(FLAGS_simclr_classifier_mindir_path).empty()) {
    std::cout << "Invalid simclr_classifier mindir path" << std::endl;
    return 1;
  }

  auto context = std::make_shared<Context>();
  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(FLAGS_device_id);
  context->MutableDeviceInfo().push_back(ascend310);
  mindspore::Graph simclr_classifier_graph;
  Serialization::Load(FLAGS_simclr_classifier_mindir_path, ModelType::kMindIR, &simclr_classifier_graph);

  Model simclr_classifier_model;
  Status ret = simclr_classifier_model.Build(GraphCell(simclr_classifier_graph), context);
  if (ret != kSuccess) {
    std::cout << "ERROR: Build simclr_classifier model failed." << std::endl;
    return 1;
  }

  std::vector<MSTensor> simclr_classifier_model_inputs = simclr_classifier_model.GetInputs();
  if (simclr_classifier_model_inputs.empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return 1;
  }

  auto input0_files = GetAllFiles(FLAGS_dataset_path);
  if (input0_files.empty()) {
    std::cout << "ERROR: no input data." << std::endl;
    return 1;
  }

  std::map<double, double> costTime_map;
  size_t size = input0_files.size();
  std::cout << "size：" << size << std::endl;

  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTimeMs;
    double endTimeMs;
    std::vector<MSTensor> model_inputs;
    std::vector<MSTensor> model_outputs;
    std::cout << "Start predict input files:" << input0_files[i] << std::endl;

    auto input0 = ReadFileToTensor(input0_files[i]);
    model_inputs.emplace_back(simclr_classifier_model_inputs[0].Name(),
                              simclr_classifier_model_inputs[0].DataType(),
                              simclr_classifier_model_inputs[0].Shape(),
                              input0.Data().get(), input0.DataSize());
    gettimeofday(&start, nullptr);
    ret = simclr_classifier_model.Predict(model_inputs, &model_outputs);
      if (ret != kSuccess) {
        std::cout << "Predict" << input0_files[i] << "failed." << std::endl;
        return 1;
      }

    gettimeofday(&end, nullptr);
    startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
    WriteResult(input0_files[i], model_outputs);
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
