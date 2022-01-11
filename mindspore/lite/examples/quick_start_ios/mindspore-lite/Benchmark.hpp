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

#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include <getopt.h>
#include <csignal>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <cfloat>
#include <random>
#include <utility>
#include <unordered_map>

#include "mindspore-lite/include/model.h"
#include "mindspore-lite/include/lite_session.h"
#include "mindspore-lite/include/errorcode.h"

constexpr float kNumRelativeTolerance = 1e-5;
constexpr float kNumAbsoluteTolerance = 1e-8;
constexpr float kNumMinAbsoluteError = 1e-5;
constexpr float kNumDefaultLoopCount = 10;
constexpr float kNumDefaultNumThread = 2;
constexpr float kNumDefaultWarmUpLoopCount = 3;
constexpr float kNumDefaultAccuracyThreshold = 0.5;
constexpr float kNumDefaultPrintLen = 50;
constexpr float kNumPercentage = 100;
constexpr float kNumMinMeanError = 0.0000001;

enum InDataType { kImage = 0, kBinary = 1 };
namespace mindspore::lite {
#ifdef ENABLE_ARM64
struct PerfResult {
  int64_t nr;
  struct {
    int64_t value;
    int64_t id;
  } values[2];
};
struct PerfCount {
  int64_t value[2];
};
#endif

struct CheckTensor {
  CheckTensor(const std::vector<size_t> &shape, const std::vector<float> &data,
              const std::vector<std::string> &strings_data = {""}) {
    this->shape = shape;
    this->data = data;
    this->strings_data = strings_data;
  }
  std::vector<size_t> shape;
  std::vector<float> data;
  std::vector<std::string> strings_data;
};

struct Flag {
  std::string model_file_;
  std::string in_data_file_;
  std::vector<std::string> input_data_list_;
  InDataType in_data_type_ = kBinary;
  std::string in_data_type_in_ = "bin";
  int cpu_bind_mode_ = HIGHER_CPU;
  // MarkPerformance
  int loop_count_ = kNumDefaultLoopCount;
  int num_threads_ = kNumDefaultNumThread;
  bool enable_fp16_ = false;
  int warm_up_loop_count_ = kNumDefaultWarmUpLoopCount;
  // MarkAccuracy
  std::string benchmark_data_file_;
  std::string benchmark_data_type_ = "FLOAT";
  float accuracy_threshold_ = kNumDefaultAccuracyThreshold;
  // Resize
  std::string resize_dims_in_;
  std::vector<std::vector<int>> resize_dims_;

  std::string device_ = "CPU";
  bool time_profiling_ = false;
  bool perf_profiling_ = false;
  std::string perf_event_ = "CYCLE";
  bool dump_profiling_ = false;
};

class Benchmark {
 public:
  explicit Benchmark(Flag *flag) : flags_(flag) {}

  virtual ~Benchmark();

  int Init();
  int RunBenchmark();

 private:
  // call GenerateInputData or ReadInputFile to init inputTensors
  int LoadInput();

  // call GenerateRandomData to fill inputTensors
  int GenerateInputData();

  int GenerateRandomData(size_t size, void *data, TypeId data_type);

  int ReadInputFile();

  int ReadCalibData();

  int ReadTensorData(std::ifstream &in_file_stream, const std::string &tensor_name, const std::vector<size_t> &dims);

  int CompareOutput();

  tensor::MSTensor *GetTensorByNameOrShape(const std::string &node_or_tensor_name, const std::vector<size_t> &dims);

  tensor::MSTensor *GetTensorByNodeShape(const std::vector<size_t> &node_shape);

  int CompareStringData(const std::string &name, tensor::MSTensor *tensor);

  int CompareDataGetTotalBiasAndSize(const std::string &name, tensor::MSTensor *tensor, float *total_bias,
                                     int *total_size);

  int InitCallbackParameter();

  int InitTimeProfilingCallbackParameter();

  int InitPerfProfilingCallbackParameter();

  int InitDumpProfilingCallbackParameter();

  int PrintResult(const std::vector<std::string> &title, const std::map<std::string, std::pair<int, float>> &result);

  void InitContext(const std::shared_ptr<Context> &context);

#ifdef ENABLE_ARM64
  int PrintPerfResult(const std::vector<std::string> &title,
                      const std::map<std::string, std::pair<int, struct PerfCount>> &result);
#endif

  int PrintInputData();

  // tensorData need to be converter first
  template <typename T>
  float CompareData(const std::string &nodeName, const std::vector<int> &msShape, const void *tensor_data) {
    const T *msTensorData = static_cast<const T *>(tensor_data);
    auto iter = this->benchmark_data_.find(nodeName);
    if (iter != this->benchmark_data_.end()) {
      std::vector<size_t> castedMSShape;
      size_t shapeSize = 1;
      for (int64_t dim : msShape) {
        castedMSShape.push_back(size_t(dim));
        shapeSize *= dim;
      }

      CheckTensor *calibTensor = iter->second;
      if (calibTensor->shape != castedMSShape) {
        std::cout << "Shape of mslite output(";
        for (auto dim : castedMSShape) {
          std::cout << dim << ",";
        }
        std::cout << ") and shape source model output(";
        for (auto dim : calibTensor->shape) {
          std::cout << dim << ",";
        }
        std::cout << ") are different";
        return RET_ERROR;
      }
      size_t errorCount = 0;
      float meanError = 0;
      std::cout << "Data of node " << nodeName << " : ";
      for (size_t j = 0; j < shapeSize; j++) {
        if (j < kNumDefaultPrintLen) {
          std::cout << static_cast<float>(msTensorData[j]) << " ";
        }

        if (std::isnan(msTensorData[j]) || std::isinf(msTensorData[j])) {
          std::cerr << "Output tensor has nan or inf data, compare fail" << std::endl;
          return RET_ERROR;
        }

        auto tolerance = kNumAbsoluteTolerance + kNumRelativeTolerance * fabs(calibTensor->data.at(j));
        auto absoluteError = std::fabs(msTensorData[j] - calibTensor->data.at(j));
        if (absoluteError > tolerance) {
          if (fabs(calibTensor->data.at(j) - 0.0f) < FLT_EPSILON) {
            if (absoluteError > kNumMinAbsoluteError) {
              meanError += absoluteError;
              errorCount++;
            } else {
              continue;
            }
          } else {
            // just assume that atol = rtol
            meanError += absoluteError / (fabs(calibTensor->data.at(j)) + FLT_MIN);
            errorCount++;
          }
        }
      }
      std::cout << std::endl;
      if (meanError > 0.0f) {
        meanError /= errorCount;
      }

      if (meanError <= kNumMinMeanError) {
        std::cout << "Mean bias of node/tensor " << nodeName << " : 0%" << std::endl;
      } else {
        std::cout << "Mean bias of node/tensor " << nodeName << " : " << meanError * kNumPercentage << "%" << std::endl;
      }
      return meanError;
    } else {
      std::cout << "%s is not in Source Model output" << nodeName.c_str();
      return RET_ERROR;
    }
  }

  template <typename T, typename Distribution>
  void FillInputData(int size, void *data, Distribution distribution) {
    if (data == nullptr) {
      std::cout << "data is nullptr.";
      return;
    }
    int elements_num = size / sizeof(T);
    (void)std::generate_n(static_cast<T *>(data), elements_num,
                          [&]() { return static_cast<T>(distribution(random_engine_)); });
  }

  int MarkPerformance();

  int MarkAccuracy();

 private:
  struct Flag *flags_;
  session::LiteSession *session_{nullptr};
  std::vector<mindspore::tensor::MSTensor *> ms_inputs_;
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> ms_outputs_;
  std::unordered_map<std::string, CheckTensor *> benchmark_data_;
  std::unordered_map<std::string, TypeId> data_type_map_{{"FLOAT", TypeId::kNumberTypeFloat},
                                                         {"INT8", TypeId::kNumberTypeInt8},
                                                         {"INT32", TypeId::kNumberTypeInt32},
                                                         {"UINT8", TypeId::kNumberTypeUInt8}};
  TypeId msCalibDataType = TypeId::kNumberTypeFloat;

  // callback parameters
  uint64_t op_begin_ = 0;
  int op_call_times_total_ = 0;
  float op_cost_total_ = 0.0f;
  std::map<std::string, std::pair<int, float>> op_times_by_type_;
  std::map<std::string, std::pair<int, float>> op_times_by_name_;
#ifdef ENABLE_ARM64
  int perf_fd = 0;
  int perf_fd2 = 0;
  float op_cost2_total_ = 0.0f;
  std::map<std::string, std::pair<int, struct PerfCount>> op_perf_by_type_;
  std::map<std::string, std::pair<int, struct PerfCount>> op_perf_by_name_;
#endif
  KernelCallBack before_call_back_ = nullptr;
  KernelCallBack after_call_back_ = nullptr;
  std::mt19937 random_engine_;
};
int RunBenchmark(Flag *flags);
int main_benchmark();
}  // namespace mindspore::lite
#endif  // MINNIE_BENCHMARK_BENCHMARK_H_
