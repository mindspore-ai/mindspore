/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_BENCHMARK_TRAIN_NET_TRAIN_BASE_H_
#define MINDSPORE_LITE_TOOLS_BENCHMARK_TRAIN_NET_TRAIN_BASE_H_

#include <getopt.h>
#include <csignal>
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
#include <nlohmann/json.hpp>
#include "include/api/model.h"
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/api/cfg.h"

#ifdef ENABLE_FP16
#include <arm_neon.h>
#endif
#include "tools/common/flag_parser.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"

#ifdef ENABLE_FP16
static __attribute__((always_inline)) inline bool MS_ISNAN_FP16(float16_t var) {
  volatile float16_t d = var;
  return d != d;
}
#endif

namespace mindspore::lite {
enum MS_API DataType { kImage = 0, kBinary = 1 };

constexpr float relativeTolerance = 1e-5;
constexpr float absoluteTolerance = 1e-8;
extern const std::unordered_map<int, std::string> kTypeIdMap;
extern const std::unordered_map<mindspore::Format, std::string> kTensorFormatMap;

constexpr const char *DELIM_SLASH = "/";
constexpr const char *DELIM_COLON = ":";
constexpr const char *DELIM_COMMA = ",";

constexpr int RET_TOO_BIG = -9;
constexpr int kFieldsToPrint = 5;
constexpr int kPrintOffset = 4;
constexpr int kDumpInputsAndOutputs = 0;
constexpr int kDumpOutputs = 2;
constexpr int kTHOUSAND = 1000;

namespace dump {
constexpr auto kConfigPath = "MINDSPORE_DUMP_CONFIG";
constexpr auto kSettings = "common_dump_settings";
constexpr auto kMode = "dump_mode";
constexpr auto kPath = "path";
constexpr auto kNetName = "net_name";
constexpr auto kInputOutput = "input_output";
constexpr auto kKernels = "kernels";
}  // namespace dump

template <typename T>
float TensorSum(const void *data, int size) {
  const T *typed_data = reinterpret_cast<const T *>(data);
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
    AddFlag(&NetTrainFlags::unified_api_, "unifiedApi", "do unified api test", false);
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
  bool unified_api_ = false;
  bool dump_tensor_data_ = false;
};

class MS_API NetTrainBase {
 public:
  explicit NetTrainBase(NetTrainFlags *flags) : flags_(flags) {}
  virtual ~NetTrainBase();

  int Init();
  int RunNetTrain();
  static float *ReadFileBuf(const std::string file, size_t *size);
  static int SetNr(std::function<int(NetTrainFlags *)> param);
  static int RunNr(NetTrainFlags *flags) {
    if (nr_cb_ != nullptr) {
      return nr_cb_(flags);
    }
    MS_LOG(WARNING) << "unified api was not tested";
    std::cout << "unified api was not tested";
    return RET_OK;
  }
  // tensorData need to be converter first
  template <typename T>
  static float CompareData(const float *refOutput, int size, const T *msTensorData) {
    size_t errorCount = 0;
    float meanError = 0;
    std::cout << "Out tensor size is: " << size << std::endl;
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
  int InitDumpConfigFromJson(std::string path);

 protected:
  // call GenerateInputData or ReadInputFile to init inputTensors
  int LoadInput();
  void CheckSum(MSTensor *tensor, const std::string &node_type, int id, const std::string &in_out);
  // call GenerateRandomData to fill inputTensors
  virtual int GenerateInputData() = 0;

  int GenerateRandomData(mindspore::MSTensor *tensor);

  std::string GenerateOutputFileName(mindspore::MSTensor *tensor, const std::string &op_name,
                                     const std::string &file_type, const size_t &idx);
  virtual int ReadInputFile() = 0;

  virtual int CreateAndRunNetwork(const std::string &filename, const std::string &bb_filename, bool is_train,
                                  int epochs, bool check_accuracy = true) = 0;

  int InitCallbackParameter();

  virtual int InitDumpTensorDataCallbackParameter() = 0;

  virtual int InitTimeProfilingCallbackParameter() = 0;

  virtual int PrintResult(const std::vector<std::string> &title,
                          const std::map<std::string, std::pair<int, float>> &result) = 0;

  template <typename T>
  std::vector<int64_t> ConverterToInt64Vector(const std::vector<T> &srcDims) {
    std::vector<int64_t> dims;
    for (auto shape : srcDims) {
      dims.push_back(static_cast<int64_t>(shape));
    }
    return dims;
  }
  virtual int MarkPerformance() = 0;
  virtual int MarkAccuracy(bool enforce_accuracy = true) = 0;
  virtual int CompareOutput() = 0;
  virtual int SaveModels() = 0;
  int CheckExecutionOfSavedModels();
  void TensorNan(const float *data, int size) {
    for (int i = 0; i < size; i++) {
      if (std::isnan(data[i])) {
        std::cout << "nan value of index=" << i << ", " << data[i] << std::endl;
        break;
      }
    }
  }
#ifdef ENABLE_FP16
  void TensorNan(float16_t *data, int size) {
    for (int i = 0; i < size; i++) {
      if (MS_ISNAN_FP16(data[i]) || std::isinf(data[i])) {
        std::cout << "nan or inf value of index=" << i << ", " << data[i] << std::endl;
        break;
      }
    }
  }
#endif
  NetTrainFlags *flags_{nullptr};
  static std::function<int(NetTrainFlags *)> nr_cb_;

  nlohmann::json dump_cfg_json_;
  std::string dump_file_output_dir_;
  std::vector<std::shared_ptr<char>> inputs_buf_;
  std::vector<size_t> inputs_size_;
  size_t batch_num_ = 0;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_BENCHMARK_TRAIN_NET_TRAIN_BASE_H_
