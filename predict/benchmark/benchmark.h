/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PREDICT_BENCHMARK_BENCHMARK_H_
#define PREDICT_BENCHMARK_BENCHMARK_H_

#include <getopt.h>
#include <signal.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "common/flag_parser.h"
#include "common/file_utils.h"
#include "common/func_utils.h"
#include "common/mslog.h"
#include "common/utils.h"
#include "include/errorcode.h"
#include "include/session.h"
#include "include/tensor.h"
#include "schema/inner/ms_generated.h"
#include "src/graph.h"
#include "src/graph_execution.h"
#include "src/op.h"

namespace mindspore {
namespace predict {
enum InDataType { kImage = 0, kBinary = 1 };

struct CheckTensor {
  CheckTensor(const std::vector<size_t> &shape, const std::vector<float> &data) {
    this->shape = shape;
    this->data = data;
  }
  std::vector<size_t> shape;
  std::vector<float> data;
};

class BenchmarkFlags : public virtual FlagParser {
 public:
  BenchmarkFlags() {
    // common
    AddFlag(&BenchmarkFlags::modelPath, "modelPath", "Input model path", "");
    AddFlag(&BenchmarkFlags::tensorDataTypeIn, "tensorDataType", "Data type of input Tensor. float", "float");
    AddFlag(&BenchmarkFlags::inDataPath, "inDataPath", "Input data path, if not set, use random input", "");
    // MarkPerformance
    AddFlag(&BenchmarkFlags::loopCount, "loopCount", "Run loop count", 10);
    AddFlag(&BenchmarkFlags::numThreads, "numThreads", "Run threads number", 2);
    AddFlag(&BenchmarkFlags::warmUpLoopCount, "warmUpLoopCount", "Run warm up loop", 3);
    // MarkAccuracy
    AddFlag(&BenchmarkFlags::calibDataPath, "calibDataPath", "Calibration data file path", "");
  }

  ~BenchmarkFlags() override = default;

 public:
  // common
  std::string modelPath;
  std::string inDataPath;
  InDataType inDataType;
  std::string inDataTypeIn;
  DataType tensorDataType;
  std::string tensorDataTypeIn;
  // MarkPerformance
  int loopCount;
  int numThreads;
  int warmUpLoopCount;
  // MarkAccuracy
  std::string calibDataPath;
};

class Benchmark {
 public:
  explicit Benchmark(BenchmarkFlags *flags) : _flags(flags) {}

  virtual ~Benchmark() = default;

  STATUS Init();
  STATUS RunBenchmark();

 private:
  // call GenerateInputData or ReadInputFile to init inputTensors
  STATUS LoadInput();

  // call GenerateRandomData to fill inputTensors
  STATUS GenerateInputData();

  STATUS GenerateRandomData(size_t size, void *data);

  STATUS ReadInputFile();

  STATUS ReadCalibData();

  STATUS CleanData();

  STATUS CompareOutput(const std::map<NODE_ID, std::vector<Tensor *>> &msOutputs);

  float CompareData(const std::string &nodeName, std::vector<int64_t> msShape, float *msTensorData);

  STATUS MarkPerformance();

  STATUS MarkAccuracy();

 private:
  BenchmarkFlags *_flags;
  std::shared_ptr<Session> session;
  Context ctx;
  std::vector<Tensor *> msInputs;
  std::map<std::string, std::vector<Tensor *>> msOutputs;
  std::unordered_map<std::string, CheckTensor *> calibData;
  std::string modelName = "";
  bool cleanData = true;

  const float US2MS = 1000.0f;
  const float percentage = 100.0f;
  const int printNum = 50;
  const float minFloatThr = 0.0000001f;

  const uint64_t maxTimeThr = 1000000;
};

int RunBenchmark(int argc, const char **argv);
}  // namespace predict
}  // namespace mindspore
#endif  // PREDICT_BENCHMARK_BENCHMARK_H_
