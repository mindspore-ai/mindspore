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
#include "include/lite_session.h"

namespace mindspore::lite {
enum MS_API DataType { kImage = 0, kBinary = 1 };

constexpr float relativeTolerance = 1e-5;
constexpr float absoluteTolerance = 1e-8;

template <typename T>
float TensorSum(void *data, int size) {
  T *typed_data = reinterpret_cast<T *>(data);
  float sum = 0.f;
  for (int i = 0; i < size; i++) {
    sum += static_cast<float>(typed_data[i]);
  }
  return sum;
}

class MS_API NetTrainFlags : public virtual FlagParser {
 public:
  NetTrainFlags() {
    // common
    AddFlag(&NetTrainFlags::model_file_, "modelFile", "Input model file", "");
    AddFlag(&NetTrainFlags::bb_model_file_, "bbModelFile", "Backboine model for transfer session", "");
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
    AddFlag(&NetTrainFlags::loss_name_, "lossName", "loss layer name", "");
    AddFlag(&NetTrainFlags::inference_file_, "inferenceFile", "MS file to export inference model", "");
    AddFlag(&NetTrainFlags::virtual_batch_, "virtualBatch", "use virtual batch", false);
    AddFlag(&NetTrainFlags::resize_dims_in_, "inputShapes",
            "Shape of input data, the format should be NHWC. e.g. 1,32,32,32:1,1,32,32,1", "");
    AddFlag(&NetTrainFlags::is_raw_mix_precision_, "isRawMixPrecision",
            "If model is mix precision export from MindSpore,please set true", false);
  }

  ~NetTrainFlags() override = default;
  void InitResizeDimsList();

 public:
  // common
  std::string model_file_;
  std::string in_data_file_;
  std::string bb_model_file_;
  std::vector<std::string> input_data_list_;
  DataType in_data_type_;
  std::string in_data_type_in_ = "bin";
  int cpu_bind_mode_ = 1;
  bool enable_fp16_ = false;
  bool virtual_batch_ = false;
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
  std::vector<std::vector<int>> resize_dims_;
  std::string loss_name_ = "";
  std::string inference_file_ = "";
  bool is_raw_mix_precision_ = false;
};

class MS_API NetTrain {
 public:
  explicit NetTrain(NetTrainFlags *flags) : flags_(flags) {}
  virtual ~NetTrain() = default;

  int Init();
  int RunNetTrain();

 private:
  // call GenerateInputData or ReadInputFile to init inputTensors
  int LoadInput(Vector<tensor::MSTensor *> *ms_inputs);
  void CheckSum(mindspore::tensor::MSTensor *tensor, std::string node_type, int id, std::string in_out);
  // call GenerateRandomData to fill inputTensors
  int GenerateInputData(std::vector<mindspore::tensor::MSTensor *> *ms_inputs);

  int GenerateRandomData(size_t size, void *data);

  int ReadInputFile(std::vector<mindspore::tensor::MSTensor *> *ms_inputs);
  int CreateAndRunNetwork(const std::string &filename, const std::string &bb_filename, int train_session, int epochs,
                          bool check_accuracy = true);

  std::unique_ptr<session::LiteSession> CreateAndRunNetworkForInference(const std::string &filename,
                                                                        const Context &context);

  std::unique_ptr<session::LiteSession> CreateAndRunNetworkForTrain(const std::string &filename,
                                                                    const std::string &bb_filename,
                                                                    const Context &context, const TrainCfg &train_cfg,
                                                                    int epochs);

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
    std::cout << std::endl;
    for (int j = 0; j < size; j++) {
      if (std::isnan(msTensorData[j]) || std::isinf(msTensorData[j])) {
        std::cerr << "Output tensor has nan or inf data, compare fail" << std::endl;
        MS_LOG(ERROR) << "Output tensor has nan or inf data, compare fail";
        return RET_ERROR;
      }

      auto tolerance = absoluteTolerance + relativeTolerance * fabs(refOutput[j]);
      auto absoluteError = std::fabs(static_cast<float>(msTensorData[j]) - refOutput[j]);
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

  int MarkPerformance(const std::unique_ptr<session::LiteSession> &session);
  int MarkAccuracy(const std::unique_ptr<session::LiteSession> &session, bool enforce_accuracy = true);
  int CompareOutput(const session::LiteSession &lite_session);
  int SaveModels(const std::unique_ptr<session::LiteSession> &session);
  int CheckExecutionOfSavedModels();
  void TensorNan(float *data, int size) {
    for (int i = 0; i < size; i++) {
      if (std::isnan(data[i])) {
        std::cout << "nan value of index=" << i << std::endl;
        break;
      }
    }
  }
  NetTrainFlags *flags_;

  // callback parameters
  uint64_t op_begin_ = 0;
  int op_call_times_total_ = 0;
  float op_cost_total_ = 0.0f;
  std::map<std::string, std::pair<int, float>> op_times_by_type_;
  std::map<std::string, std::pair<int, float>> op_times_by_name_;

  mindspore::KernelCallBack before_call_back_;
  mindspore::KernelCallBack after_call_back_;
};

int MS_API RunNetTrain(int argc, const char **argv);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_BENCHMARK_TRAIN_NET_TRAIN_H_
