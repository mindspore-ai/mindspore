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
#include <string>
#include <vector>
#include <memory>
#include "tools/common/flag_parser.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#include "schema/model_generated.h"
#include "include/model.h"
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
    AddFlag(&BenchmarkFlags::inDataTypeIn, "inDataType", "Input data type. img | bin", "bin");
    AddFlag(&BenchmarkFlags::omModelPath, "omModelPath", "OM model path, only required when device is NPU", "");
    AddFlag(&BenchmarkFlags::device, "device", "CPU | NPU | GPU", "CPU");
    AddFlag(&BenchmarkFlags::cpuBindMode, "cpuBindMode",
            "Input -1 for MID_CPU, 1 for HIGHER_CPU, 0 for NO_BIND, defalut value: 1", 1);
    // MarkPerformance
    AddFlag(&BenchmarkFlags::loopCount, "loopCount", "Run loop count", 10);
    AddFlag(&BenchmarkFlags::numThreads, "numThreads", "Run threads number", 2);
    AddFlag(&BenchmarkFlags::warmUpLoopCount, "warmUpLoopCount", "Run warm up loop", 3);
    // MarkAccuracy
    AddFlag(&BenchmarkFlags::calibDataPath, "calibDataPath", "Calibration data file path", "");
    AddFlag(&BenchmarkFlags::accuracyThreshold, "accuracyThreshold", "Threshold of accuracy", 0.5);
    // Resize
    AddFlag(&BenchmarkFlags::resizeDimsIn, "resizeDims", "Dims to resize to", "");
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
  std::string inDataTypeIn;
  int cpuBindMode = 1;
  // MarkPerformance
  int loopCount;
  int numThreads;
  int warmUpLoopCount;
  // MarkAccuracy
  std::string calibDataPath;
  float accuracyThreshold;
  // Resize
  std::string resizeDimsIn;
  std::vector<std::vector<int64_t>> resizeDims;

  std::string omModelPath;
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

  float CompareData(const std::string &nodeName, std::vector<int> msShape, float *msTensorData);

  int MarkPerformance();

  int MarkAccuracy();

 private:
  BenchmarkFlags *_flags;
  session::LiteSession *session;
  std::vector<mindspore::tensor::MSTensor *> msInputs;
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> msOutputs;
  std::unordered_map<std::string, CheckTensor *> calibData;
  bool cleanData = true;
};

int MS_API RunBenchmark(int argc, const char **argv);
}  // namespace mindspore::lite
#endif  // MINNIE_BENCHMARK_BENCHMARK_H_
