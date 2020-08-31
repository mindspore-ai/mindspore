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

#ifndef MINNIE_BENCHMARK_BENCHMARK_H_
#define MINNIE_BENCHMARK_BENCHMARK_H_

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
#include "include/model.h"
#include "tools/common/flag_parser.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#include "include/lite_session.h"
#include "include/inference.h"

namespace mindspore::lite {
enum MS_API InDataType { kImage = 0, kBinary = 1 };

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

class MS_API BenchmarkFlags : public virtual FlagParser {
 public:
  BenchmarkFlags() {
    // common
    AddFlag(&BenchmarkFlags::modelPath, "modelPath", "Input model path", "");
    AddFlag(&BenchmarkFlags::inDataPath, "inDataPath", "Input data path, if not set, use random input", "");
    AddFlag(&BenchmarkFlags::device, "device", "CPU | GPU", "CPU");
    AddFlag(&BenchmarkFlags::cpuBindMode, "cpuBindMode",
            "Input -1 for MID_CPU, 1 for HIGHER_CPU, 0 for NO_BIND, defalut value: 1", 1);
    // MarkPerformance
    AddFlag(&BenchmarkFlags::loopCount, "loopCount", "Run loop count", 10);
    AddFlag(&BenchmarkFlags::numThreads, "numThreads", "Run threads number", 2);
    AddFlag(&BenchmarkFlags::fp16Priority, "fp16Priority", "Priority float16", false);
    AddFlag(&BenchmarkFlags::warmUpLoopCount, "warmUpLoopCount", "Run warm up loop", 3);
    // MarkAccuracy
    AddFlag(&BenchmarkFlags::calibDataPath, "calibDataPath", "Calibration data file path", "");
    AddFlag(&BenchmarkFlags::calibDataType, "calibDataType", "Calibration data type. FLOAT | INT32 | INT8", "FLOAT");
    AddFlag(&BenchmarkFlags::accuracyThreshold, "accuracyThreshold", "Threshold of accuracy", 0.5);
  }

  ~BenchmarkFlags() override = default;

  void InitInputDataList();

  void InitResizeDimsList();

 public:
  // common
  std::string modelPath;
  std::string inDataPath;
  std::vector<std::string> input_data_list;
  InDataType inDataType;
  std::string inDataTypeIn = "bin";
  int cpuBindMode = 1;
  // MarkPerformance
  int loopCount;
  int numThreads;
  bool fp16Priority;
  int warmUpLoopCount;
  // MarkAccuracy
  std::string calibDataPath;
  std::string calibDataType;
  float accuracyThreshold;
  // Resize
  std::string resizeDimsIn = "";
  std::vector<std::vector<int64_t>> resizeDims;

  std::string device;
};

class MS_API Benchmark {
 public:
  explicit Benchmark(BenchmarkFlags *flags) : _flags(flags) {}

  virtual ~Benchmark();

  int Init();
  int RunBenchmark(const std::string &deviceType = "NPU");
  //  int RunNPUBenchmark();

 private:
  // call GenerateInputData or ReadInputFile to init inputTensors
  int LoadInput();

  // call GenerateRandomData to fill inputTensors
  int GenerateInputData();

  int GenerateRandomData(size_t size, void *data);

  int ReadInputFile();

  int ReadCalibData();

  int CompareOutput();

  template <typename T>
  void PrintInputData(tensor::MSTensor *input) {
    MS_ASSERT(input != nullptr);
    static int i = 0;
    auto inData = reinterpret_cast<T *>(input->MutableData());
    std::cout << "InData" << i++ << ": ";
//    int printSize = std::min(20, input->ElementsNum());
    for (size_t j = 0; j < 20; j++) {
      std::cout << static_cast<float >(inData[j]) << " ";
    }
    std::cout << std::endl;
  }

  // tensorData need to be converter first
  template <typename T>
  float CompareData(const std::string &nodeName, std::vector<int> msShape, T *msTensorData) {
    auto iter = this->calibData.find(nodeName);
    if (iter != this->calibData.end()) {
      std::vector<size_t> castedMSShape;
      size_t shapeSize = 1;
      for (int64_t dim : msShape) {
        castedMSShape.push_back(size_t(dim));
        shapeSize *= dim;
      }

      CheckTensor *calibTensor = iter->second;
      if (calibTensor->shape != castedMSShape) {
        std::ostringstream oss;
        oss << "Shape of mslite output(";
        for (auto dim : castedMSShape) {
          oss << dim << ",";
        }
        oss << ") and shape source model output(";
        for (auto dim : calibTensor->shape) {
          oss << dim << ",";
        }
        oss << ") are different";
        std::cerr << oss.str() << std::endl;
        MS_LOG(ERROR) << oss.str().c_str();
        return RET_ERROR;
      }
      size_t errorCount = 0;
      float meanError = 0;
      std::cout << "Data of node " << nodeName << " : ";
      for (size_t j = 0; j < shapeSize; j++) {
        if (j < 50) {
          std::cout << static_cast<float>(msTensorData[j]) << " ";
        }

        if (std::isnan(msTensorData[j]) || std::isinf(msTensorData[j])) {
          std::cerr << "Output tensor has nan or inf data, compare fail" << std::endl;
          MS_LOG(ERROR) << "Output tensor has nan or inf data, compare fail";
          return RET_ERROR;
        }

        auto tolerance = absoluteTolerance + relativeTolerance * fabs(calibTensor->data.at(j));
        auto absoluteError = std::fabs(msTensorData[j] - calibTensor->data.at(j));
        if (absoluteError > tolerance) {
          // just assume that atol = rtol
          meanError += absoluteError / (fabs(calibTensor->data.at(j)) + FLT_MIN);
          errorCount++;
        }
      }
      std::cout << std::endl;
      if (meanError > 0.0f) {
        meanError /= errorCount;
      }

      if (meanError <= 0.0000001) {
        std::cout << "Mean bias of node " << nodeName << " : 0%" << std::endl;
      } else {
        std::cout << "Mean bias of node " << nodeName << " : " << meanError * 100 << "%" << std::endl;
      }
      return meanError;
    } else {
      MS_LOG(INFO) << "%s is not in Source Model output", nodeName.c_str();
      return RET_ERROR;
    }
  }

  int MarkPerformance();

  int MarkAccuracy();

 private:
  BenchmarkFlags *_flags;
  session::LiteSession *session;
  std::vector<mindspore::tensor::MSTensor *> msInputs;
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> msOutputs;
  std::unordered_map<std::string, CheckTensor *> calibData;
  std::unordered_map<std::string, TypeId> dataTypeMap{
    {"FLOAT", TypeId::kNumberTypeFloat}, {"INT8", TypeId::kNumberTypeInt8}, {"INT32", TypeId::kNumberTypeInt32}};
//  TypeId msInputBinDataType = TypeId::kNumberTypeFloat;
  TypeId msCalibDataType = TypeId::kNumberTypeFloat;
};

int MS_API RunBenchmark(int argc, const char **argv);
}  // namespace mindspore::lite
#endif  // MINNIE_BENCHMARK_BENCHMARK_H_
