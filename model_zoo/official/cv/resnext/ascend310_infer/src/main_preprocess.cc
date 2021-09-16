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
#include "inc/utils.h"

using mindspore::Context;
using mindspore::GraphCell;
using mindspore::Model;
using mindspore::ModelType;
using mindspore::MSTensor;
using mindspore::Serialization;
using mindspore::Status;

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_string(image_path, ".", "image path");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  auto context = std::make_shared<Context>();
  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(FLAGS_device_id);
  context->MutableDeviceInfo().push_back(ascend310);
  mindspore::Graph graph;
  Serialization::Load(FLAGS_mindir_path, ModelType::kMindIR, &graph);
  Model model;
  Status ret = model.Build(GraphCell(graph), context);
  if (ret.IsError()) {
    std::cout << "ERROR: Build failed." << std::endl;
    return 1;
  }

  std::cout << "Check if data preprocess exists: " << model.HasPreprocess() << std::endl;

  // way 1, construct a common MSTensor
  std::vector<MSTensor> inputs1 = {ReadFileToTensor(FLAGS_image_path)};
  std::vector<MSTensor> outputs1;

  ret = model.PredictWithPreprocess(inputs1, &outputs1);
  if (ret.IsError()) {
    std::cout << "ERROR: Predict failed." << std::endl;
    return 1;
  }

  std::ofstream o1("result1.txt", std::ios::out);
  o1.write(reinterpret_cast<const char *>(outputs1[0].MutableData()), std::streamsize(outputs1[0].DataSize()));

  // way 2, construct a pointer of MSTensor, be careful of destroy
  MSTensor *tensor = MSTensor::CreateImageTensor(FLAGS_image_path);
  std::vector<MSTensor> inputs2 = {*tensor};
  MSTensor::DestroyTensorPtr(tensor);
  std::vector<MSTensor> outputs2;

  ret = model.PredictWithPreprocess(inputs2, &outputs2);
  if (ret.IsError()) {
    std::cout << "ERROR: Predict failed." << std::endl;
    return 1;
  }

  std::ofstream o2("result2.txt", std::ios::out);
  o2.write(reinterpret_cast<const char *>(outputs2[0].MutableData()), std::streamsize(outputs2[0].DataSize()));

  // way 3, split preprocess and predict
  std::vector<MSTensor> inputs3 = {ReadFileToTensor(FLAGS_image_path)};
  std::vector<MSTensor> outputs3;

  ret = model.Preprocess(inputs3, &outputs3);
  if (ret.IsError()) {
    std::cout << "ERROR: Preprocess failed." << std::endl;
    return 1;
  }

  std::vector<MSTensor> outputs4;
  ret = model.Predict(outputs3, &outputs4);
  if (ret.IsError()) {
    std::cout << "ERROR: Preprocess failed." << std::endl;
    return 1;
  }

  std::ofstream o3("result3.txt", std::ios::out);
  o3.write(reinterpret_cast<const char *>(outputs4[0].MutableData()), std::streamsize(outputs4[0].DataSize()));

  // check shape
  auto shape = outputs1[0].Shape();
  std::cout << "Output Shape: " << std::endl;
  for (auto s : shape) {
    std::cout << s << ", ";
  }
  std::cout << std::endl;

  return 0;
}
