/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_BENCHMARK_TRAIN_NET_TRAIN_H_
#define MINDSPORE_LITE_TOOLS_BENCHMARK_TRAIN_NET_TRAIN_H_

#include <getopt.h>
#include <signal.h>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <map>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <cfloat>
#include <utility>
#include <algorithm>
#include "tools/common/flag_parser.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#include "include/train/train_session.h"

namespace mindspore::lite {
enum MS_API DataType { kImage = 0, kBinary = 1 };

constexpr float relativeTolerance = 1e-5;
constexpr float absoluteTolerance = 1e-8;

struct MS_API CheckTensor {
  CheckTensor(const std::vector<size_t> &shape, const std::vector<float> &data) {
    this->shape = shape;
    this->data = data;
  }
  std::vector<size_t> shape;
  std::vector<float> data;
};

class MS_API NetTrainFlags : public virtual FlagParser {
 public:
  NetTrainFlags() {
    // common
    AddFlag(&NetTrainFlags::model_file_, "modelFile", "Input model file", "");
    AddFlag(&NetTrainFlags::in_data_file_, "inDataFile", "Input data file, if not set, use random input", "");
    // MarkPerformance
    AddFlag(&NetTrainFlags::warm_up_loop_count_, "warmUpLoopCount", "Run warm up loop", 0);
    AddFlag(&NetTrainFlags::time_profiling_, "timeProfiling", "Run time profiling", false);
    AddFlag(&NetTrainFlags::epochs_, "epochs", "Number of training epochs to run", 1);
    AddFlag(&NetTrainFlags::num_threads_, "numThreads", "Run threads number", 1);
    // MarkAccuracy
    AddFlag(&NetTrainFlags::data_file_, "expectedDataFile", "Expected results data file path", "");
    AddFlag(&NetTrainFlags::export_file_, "exportFile", "MS File to export trained model into", "");
    AddFlag(&NetTrainFlags::accuracy_threshold_, "accuracyThreshold", "Threshold of accuracy", 0.5);
    AddFlag(&NetTrainFlags::layer_checksum_, "layerCheckSum", "layer output checksum print (debug)", false);
    AddFlag(&NetTrainFlags::enable_fp16_, "enableFp16", "Enable float16", false);
  }

  ~NetTrainFlags() override = default;

  void InitInputDataList();

  void InitResizeDimsList();

 public:
  // common
  std::string model_file_;
  std::string in_data_file_;
  std::vector<std::string> input_data_list_;
  DataType in_data_type_;
  std::string in_data_type_in_ = "bin";
  int cpu_bind_mode_ = 1;
  bool enable_fp16_ = false;
  // MarkPerformance
  int num_threads_ = 1;
  int warm_up_loop_count_ = 0;
  bool time_profiling_;
  int epochs_ = 1;
  // MarkAccuracy
  std::string data_file_;
  std::string data_type_ = "FLOAT";
  float accuracy_threshold_;
  // Resize
  std::string export_file_ = "";
  std::string resize_dims_in_ = "";
  bool layer_checksum_ = false;
  std::vector<std::vector<int64_t>> resize_dims_;
};

class MS_API NetTrain {
 public:
  explicit NetTrain(NetTrainFlags *flags) : flags_(flags) {}

  virtual ~NetTrain();

  int Init();
  int RunNetTrain();
  int RunExportedNet();

 private:
  // call GenerateInputData or ReadInputFile to init inputTensors
  int LoadInput();

  // call GenerateRandomData to fill inputTensors
  int GenerateInputData();

  int GenerateRandomData(size_t size, void *data);

  int ReadInputFile();

  int CompareOutput();

  int InitCallbackParameter();

  int PrintResult(const std::vector<std::string> &title, const std::map<std::string, std::pair<int, float>> &result);

  template <typename T>
  void PrintInputData(tensor::MSTensor *input) {
    MS_ASSERT(input != nullptr);
    static int i = 0;
    auto inData = reinterpret_cast<T *>(input->MutableData());
    size_t tensorSize = input->ElementsNum();
    size_t len = (tensorSize < 20) ? tensorSize : 20;
    std::cout << "InData" << i++ << ": ";
    for (size_t j = 0; j < len; j++) {
      std::cout << inData[j] << " ";
    }
    std::cout << std::endl;
  }

  // tensorData need to be converter first
  template <typename T>
  float CompareData(const float *refOutput, int size, T *msTensorData) {
    size_t errorCount = 0;
    float meanError = 0;
    std::cout << "Data of model output: ";
    for (int j = 0; j < std::min(50, size); j++) {
      std::cout << static_cast<float>(msTensorData[j]) << " ";
    }
    std::cout << std::endl;
    std::cout << "Data of Ref output  : ";
    for (int j = 0; j < std::min(50, size); j++) {
      std::cout << refOutput[j] << " ";
    }
    for (int j = 0; j < size; j++) {
      if (std::isnan(msTensorData[j]) || std::isinf(msTensorData[j])) {
        std::cerr << "Output tensor has nan or inf data, compare fail" << std::endl;
        MS_LOG(ERROR) << "Output tensor has nan or inf data, compare fail";
        return RET_ERROR;
      }

      auto tolerance = absoluteTolerance + relativeTolerance * fabs(refOutput[j]);
      auto absoluteError = std::fabs(msTensorData[j] - refOutput[j]);
      if (absoluteError > tolerance) {
        if (fabs(refOutput[j]) == 0) {
          if (absoluteError > 1e-5) {
            meanError += absoluteError;
            errorCount++;
          } else {
            continue;
          }
        } else {
          // just assume that atol = rtol
          meanError += absoluteError / (fabs(refOutput[j]) + FLT_MIN);
          errorCount++;
        }
      }
    }
    std::cout << std::endl;
    if (meanError > 0.0f) {
      meanError /= errorCount;
    }

    if (meanError <= 0.0000001) {
      std::cout << "Mean bias of tensor: 0%" << std::endl;
    } else {
      std::cout << "Mean bias of tensor: " << meanError * 100 << "%" << std::endl;
    }
    return meanError;
  }

  int MarkPerformance();

  int MarkAccuracy();

 private:
  NetTrainFlags *flags_;
  session::TrainSession *session_ = nullptr;
  std::vector<mindspore::tensor::MSTensor *> ms_inputs_;
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> ms_outputs_;
  std::unordered_map<std::string, CheckTensor *> data_;
  std::unordered_map<std::string, TypeId> data_type_map_{{"FLOAT", TypeId::kNumberTypeFloat},
                                                         {"INT32", TypeId::kNumberTypeInt32}};

  // callback parameters
  uint64_t op_begin_ = 0;
  int op_call_times_total_ = 0;
  float op_cost_total_ = 0.0f;
  std::map<std::string, std::pair<int, float>> op_times_by_type_;
  std::map<std::string, std::pair<int, float>> op_times_by_name_;

  mindspore::KernelCallBack before_call_back_;
  mindspore::KernelCallBack after_call_back_;
  bool layer_checksum_ = false;
};

int MS_API RunNetTrain(int argc, const char **argv);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_BENCHMARK_TRAIN_NET_TRAIN_H_
