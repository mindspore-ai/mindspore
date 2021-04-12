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
#include "include/api/serialization.h"
#include "include/api/types.h"
#include "include/minddata/dataset/include/vision.h"
#include "include/minddata/dataset/include/execute.h"
#include "minddata/dataset/include/vision.h"

#include "inc/utils.h"

using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::Graph;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::dataset::Execute;
using mindspore::dataset::vision::Decode;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::HWC2CHW;

using mindspore::dataset::transforms::TypeCast;

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(precision_mode, "allow_fp32_to_fp16", "precision mode");
DEFINE_string(op_select_impl_mode, "", "op select impl mode");
DEFINE_string(aipp_path, "", "aipp config file");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  auto context = std::make_shared<Context>();
  auto ascend310_info = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310_info->SetDeviceID(FLAGS_device_id);
  ascend310_info->SetInsertOpConfigPath({FLAGS_aipp_path});
  context->MutableDeviceInfo().push_back(ascend310_info);

  Graph graph;
  Status ret = Serialization::Load(FLAGS_mindir_path, ModelType::kMindIR, &graph);
  if (ret != kSuccess) {
    std::cout << "Load model failed." << std::endl;
    return 1;
  }

  Model model;
  ret = model.Build(GraphCell(graph), context);
  if (ret != kSuccess) {
    std::cout << "ERROR: Build failed." << std::endl;
    return 1;
  }

  std::vector<MSTensor> modelInputs = model.GetInputs();

  auto allFiles = GetAllFiles(FLAGS_dataset_path);
  if (allFiles.empty()) {
    std::cout << "ERROR: no input data." << std::endl;
    return 1;
  }

  Execute compose({std::shared_ptr<Decode>(new Decode()),
                   std::shared_ptr<Resize>(new Resize({32, 100})),
                   std::shared_ptr<Normalize>(new Normalize({127.5, 127.5, 127.5},
                                                                            {127.5, 127.5, 127.5})),
                   std::shared_ptr<HWC2CHW>(new HWC2CHW())});
  Execute composeCast(std::shared_ptr<TypeCast>(new TypeCast("float16")));

  struct timeval start;
  struct timeval end;
  double startTime_ms;
  double endTime_ms;
  std::map<double, double> costTime_map;
  size_t size = allFiles.size();

  for (size_t i = 0; i < size; ++i) {
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    std::cout << "Start predict input files:" << allFiles[i] << std::endl;
    std::string suffix = allFiles[i].substr(allFiles[i].rfind("."));
    if (suffix != ".jpg" && suffix != ".png" && suffix != ".JPG" && suffix != ".PNG") {
      std::cout << "wrong file format: " << allFiles[i] << std::endl;
      continue;
    }

    mindspore::MSTensor img;
    compose(ReadFileToTensor(allFiles[i]), &img);

    inputs.emplace_back(modelInputs[0].Name(), modelInputs[0].DataType(), modelInputs[0].Shape(),
                        img.Data().get(), img.DataSize());

    gettimeofday(&start, NULL);
    ret = model.Predict(inputs, &outputs);
    gettimeofday(&end, NULL);
    if (ret != kSuccess) {
      std::cout << "Predict " << allFiles[i] << " failed." << std::endl;
      return 1;
    }
    startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
    WriteResult(allFiles[i], outputs);
  }
    double average = 0.0;
    int infer_cnt = 0;

    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        average += diff;
        infer_cnt++;
    }
    average = average/infer_cnt;

    std::stringstream timeCost;
    timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << infer_cnt << std::endl;
    std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << infer_cnt << std::endl;
    std::string file_name = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
    file_stream << timeCost.str();
    file_stream.close();
    costTime_map.clear();
  return 0;
}
