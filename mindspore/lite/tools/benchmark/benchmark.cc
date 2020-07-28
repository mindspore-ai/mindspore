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

#include "tools/benchmark/benchmark.h"
#define __STDC_FORMAT_MACROS
#include <cinttypes>
#undef __STDC_FORMAT_MACROS
#include <cmath>
#include <algorithm>
#include <utility>
#include "src/common/common.h"
#include "include/ms_tensor.h"
#include "include/context.h"

namespace mindspore {
namespace lite {
int Benchmark::GenerateRandomData(size_t size, void *data) {
  MS_ASSERT(data != nullptr);
  char *castedData = static_cast<char *>(data);
  for (size_t i = 0; i < size; i++) {
    castedData[i] = static_cast<char>(i);
  }
  return 0;
}

int Benchmark::GenerateInputData() {
  for (auto tensor : msInputs) {
    MS_ASSERT(tensor != nullptr);
    auto inputData = tensor->MutableData();
    if (inputData == nullptr) {
      MS_LOG(ERROR) << "MallocData for inTensor failed";
      return RET_ERROR;
    }
    MS_ASSERT(tensor->GetData() != nullptr);
    auto tensorByteSize = tensor->Size();
    auto status = GenerateRandomData(tensorByteSize, inputData);
    if (status != 0) {
      MS_LOG(ERROR) << "GenerateRandomData for inTensor failed %d" << status;
      return status;
    }
  }
  return 0;
}

int Benchmark::LoadInput() {
  if (_flags->inDataPath.empty()) {
    auto status = GenerateInputData();
    if (status != 0) {
      MS_LOG(ERROR) << "Generate input data error " << status;
      return status;
    }
  } else {
    auto status = ReadInputFile();
    if (status != 0) {
      MS_LOG(ERROR) << "ReadInputFile error, " << status;
      return status;
    }
  }
  return 0;
}

int Benchmark::ReadInputFile() {
  if (msInputs.empty()) {
    return 0;
  }

  if (this->_flags->inDataType == kImage) {
    //    int cvFlags;
    //    if (inTensor->Channel() == 3) {
    //      cvFlags = 0;  // cv::IMREAD_COLOR;
    //    } else if (inTensor->Channel() == 1) {
    //      cvFlags = 1;  // cv::IMREAD_GRAYSCALE;
    //    } else {
    //      MS_LOG(ERROR) << "Image mode only support imgChannel == 1 or 3, imgChannel : %lld", (long
    //      long)inTensor->Channel(); return RET_PARAM_INVALID;
    //    }
    // todo fill inTensor->GetData()
  } else {
    for (auto i = 0; i < _flags->input_data_list.size(); i++) {
      auto cur_tensor = msInputs.at(i);
      MS_ASSERT(cur_tensor != nullptr);
      size_t size;
      char *binBuf = ReadFile(_flags->input_data_list[i].c_str(), &size);
      auto tensorDataSize = cur_tensor->Size();
      if (size != tensorDataSize) {
        MS_LOG(ERROR) << "Input binary file size error, required: %zu, in fact: %zu" << tensorDataSize << size;
        return RET_ERROR;
      }
      auto inputData = cur_tensor->MutableData();
      memcpy(inputData, binBuf, tensorDataSize);
    }
  }
  return 0;
}

// calibData is FP32
int Benchmark::ReadCalibData() {
  const char *calibDataPath = _flags->calibDataPath.c_str();
  // read calib data
  std::ifstream inFile(calibDataPath);
  if (!inFile.good()) {
    MS_LOG(ERROR) << "file: " << calibDataPath << " is not exist";
    return 1;
  }

  if (!inFile.is_open()) {
    MS_LOG(ERROR) << "file: " << calibDataPath << " open failed";
    inFile.close();
    return 1;
  }

  std::string line;

  MS_LOG(INFO) << "Start reading calibData file";
  std::string tensorName;
  while (!inFile.eof()) {
    getline(inFile, line);
    std::stringstream stringLine1(line);
    size_t dim = 0;
    stringLine1 >> tensorName >> dim;
    std::vector<size_t> dims;
    size_t shapeSize = 1;
    for (size_t i = 0; i < dim; i++) {
      size_t tmpDim;
      stringLine1 >> tmpDim;
      dims.push_back(tmpDim);
      shapeSize *= tmpDim;
    }

    getline(inFile, line);
    std::stringstream stringLine2(line);
    std::vector<float> tensorData;
    for (size_t i = 0; i < shapeSize; i++) {
      float tmpData;
      stringLine2 >> tmpData;
      tensorData.push_back(tmpData);
    }

    auto *checkTensor = new CheckTensor(dims, tensorData);
    this->calibData.insert(std::make_pair(tensorName, checkTensor));
  }
  inFile.close();
  MS_LOG(INFO) << "Finish reading calibData file";
  return 0;
}

// tensorData need to be converter first
float Benchmark::CompareData(const std::string &nodeName, std::vector<int> msShape, float *msTensorData) {
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
      MS_LOG(ERROR) << "%s", oss.str().c_str();
      return -1;
    }
    size_t errorCount = 0;
    float meanError = 0;
    std::cout << "Data of node " << nodeName << " : ";
    for (size_t j = 0; j < shapeSize; j++) {
      if (j < 50) {
        std::cout << msTensorData[j] << " ";
      }

      auto tolerance = absoluteTolerance + relativeTolerance * fabs(calibTensor->data.at(j));
      auto absoluteError = std::fabs(msTensorData[j] - calibTensor->data.at(j));
      if (absoluteError > tolerance) {
        // just assume that atol = rtol
        meanError += absoluteError / (fabs(calibTensor->data.at(j)) + 1);
        errorCount++;
      }
    }
    std::cout << std::endl;
    if (meanError > 0.0f) {
      errorCount = 0;
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
    return -1;
  }
}

int Benchmark::CompareOutput() {
  std::cout << "================ Comparing Output data ================" << std::endl;
  float totalBias = 0;
  int totalSize = 0;
  bool hasError = false;
  for (const auto &calibTensor : calibData) {
    std::string nodeName = calibTensor.first;
    auto tensors = session->GetOutputsByName(nodeName);
    for (auto tensor : tensors) {
      MS_ASSERT(tensor->GetDataType() == DataType_DT_FLOAT);
      MS_ASSERT(tensor->GetData() != nullptr);
      float bias = CompareData(nodeName, tensor->shape(), static_cast<float *>(tensor->MutableData()));
      if (bias >= 0) {
        totalBias += bias;
        totalSize++;
      } else {
        hasError = true;
        break;
      }
    }
  }

  if (!hasError) {
    float meanBias;
    if (totalSize != 0) {
      meanBias = totalBias / totalSize * 100;
    } else {
      meanBias = 0;
    }

    std::cout << "Mean bias of all nodes: " << meanBias << "%" << std::endl;
    std::cout << "=======================================================" << std::endl << std::endl;

    if (meanBias > this->_flags->accuracyThreshold) {
      MS_LOG(ERROR) << "Mean bias of all nodes is too big: " << meanBias << "%%";
      return 1;
    } else {
      return 0;
    }
  } else {
    MS_LOG(ERROR) << "Error in CompareData";
    std::cout << "=======================================================" << std::endl << std::endl;
    return 1;
  }
}

int Benchmark::MarkPerformance() {
  MS_LOG(INFO) << "Running warm up loops...";
  for (int i = 0; i < _flags->warmUpLoopCount; i++) {
    auto status = session->RunGraph();
    if (status != 0) {
      MS_LOG(ERROR) << "Inference error %d" << status;
      return status;
    }
  }

  MS_LOG(INFO) << "Running benchmark loops...";
  uint64_t timeMin = 1000000;
  uint64_t timeMax = 0;
  uint64_t timeAvg = 0;

  for (int i = 0; i < _flags->loopCount; i++) {
    session->BindThread(true);
    auto start = GetTimeUs();
    auto status = session->RunGraph();
    if (status != 0) {
      MS_LOG(ERROR) << "Inference error %d" << status;
      return status;
    }

    auto end = GetTimeUs();
    auto time = end - start;
    timeMin = std::min(timeMin, time);
    timeMax = std::max(timeMax, time);
    timeAvg += time;

    session->BindThread(false);
  }
  if (_flags->loopCount > 0) {
    timeAvg /= _flags->loopCount;
    //    MS_LOG(INFO) << "CSV:%s:%d:%f:%f:%f\n", _flags->modelPath.substr(_flags->modelPath.find_last_of(DELIM_SLASH) +
    //    1).c_str(),
    //            _flags->numThreads, timeMin / 1000.0f, timeMax / 1000.0f, timeAvg / 1000.0f);
    //    MS_LOG(INFO) <<"Modle = %s, numThreads = %d, MinRunTime = %f ms, MaxRuntime = %f ms, AvgRunTime = %f ms",
    //            _flags->modelPath.substr(_flags->modelPath.find_last_of(DELIM_SLASH) + 1).c_str(), _flags->numThreads,
    //            timeMin / 1000.0f, timeMax / 1000.0f, timeAvg / 1000.0f);

    printf("CSV:%s:%d:%f:%f:%f\n", _flags->modelPath.substr(_flags->modelPath.find_last_of(DELIM_SLASH) + 1).c_str(),
           _flags->numThreads, timeMin / 1000.0f, timeMax / 1000.0f, timeAvg / 1000.0f);
    printf("Modle = %s, numThreads = %d, MinRunTime = %f ms, MaxRuntime = %f ms, AvgRunTime = %f ms\n",
           _flags->modelPath.substr(_flags->modelPath.find_last_of(DELIM_SLASH) + 1).c_str(), _flags->numThreads,
           timeMin / 1000.0f, timeMax / 1000.0f, timeAvg / 1000.0f);
  }
  return 0;
}

int Benchmark::MarkAccuracy() {
  MS_LOG(INFO) << "MarkAccuracy";
  for (size_t i = 0; i < msInputs.size(); i++) {
    auto inData = reinterpret_cast<float *>(msInputs.at(i)->MutableData());
    std::cout << "InData" << i << ": ";
    for (size_t j = 0; j < 20; j++) {
      std::cout << inData[j] << " ";
    }
    std::cout << std::endl;
  }
  auto status = session->RunGraph();
  if (status != 0) {
    MS_LOG(ERROR) << "Inference error %d" << status;
    return status;
  }

  ReadCalibData();
  if (cleanData) {
    for (auto &msOutput : msOutputs) {
      for (auto &outputTensor : msOutput.second) {
        delete outputTensor;
      }
    }
    msOutputs.clear();
  }
  return 0;
}

int Benchmark::RunBenchmark(const std::string &deviceType) {
  auto startPrepareTime = GetTimeUs();
  // Load graph
  std::string modelName = _flags->modelPath.substr(_flags->modelPath.find_last_of(DELIM_SLASH) + 1);

  MS_LOG(INFO) << "start reading model file";
  size_t size = 0;
  char *graphBuf = ReadFile(_flags->modelPath.c_str(), &size);
  if (graphBuf == nullptr) {
    MS_LOG(ERROR) << "Load graph failed while running %s", modelName.c_str();
    return 1;
  }
  auto model = lite::Model::Import(graphBuf, size);
  auto context = new lite::Context;
  if (_flags->device == "CPU") {
    context->deviceCtx.type = lite::DT_CPU;
  } else {
    context->deviceCtx.type = lite::DT_NPU;
  }

  if (_flags->cpuBindMode == -1) {
    context->cpuBindMode = MID_CPU;
  } else if (_flags->cpuBindMode == 0) {
    context->cpuBindMode = HIGHER_CPU;
  } else {
    context->cpuBindMode = NO_BIND;
  }
  context->threadNum = _flags->numThreads;
  session = session::LiteSession::CreateSession(context);
  auto ret = session->CompileGraph(model.get());
  if (ret != RET_OK) {
    return ret;
  }
  msInputs = session->GetInputs();
  auto endPrepareTime = GetTimeUs();
#if defined(__arm__)
  MS_LOG(INFO) << "PrepareTime = %lld ms, " << (endPrepareTime - startPrepareTime) / 1000;
  printf("PrepareTime = %lld ms, ", (endPrepareTime - startPrepareTime) / 1000);
#else
  MS_LOG(INFO) << "PrepareTime = %ld ms, " << (endPrepareTime - startPrepareTime) / 1000;
  printf("PrepareTime = %ld ms, ", (endPrepareTime - startPrepareTime) / 1000);
#endif

  // Load input
  MS_LOG(INFO) << "start generate input data";
  auto status = LoadInput();
  if (status != 0) {
    MS_LOG(ERROR) << "Generate input data error";
    return status;
  }
  if (!_flags->calibDataPath.empty()) {
    status = MarkAccuracy();
    if (status != 0) {
      MS_LOG(ERROR) << "Run MarkAccuracy error: %d" << status;
      return status;
    }
  } else {
    status = MarkPerformance();
    if (status != 0) {
      MS_LOG(ERROR) << "Run MarkPerformance error: %d" << status;
      return status;
    }
  }

  if (cleanData) {
    for (auto &msInput : msInputs) {
      delete msInput;
    }
    msInputs.clear();
    for (auto &data : calibData) {
      data.second->shape.clear();
      data.second->data.clear();
      delete data.second;
    }
    calibData.clear();
  }

  delete graphBuf;
  return 0;
}

void BenchmarkFlags::InitInputDataList() {
  char *input_list = new char[this->inDataPath.length() + 1];
  snprintf(input_list, this->inDataPath.length() + 1, "%s", this->inDataPath.c_str());
  char *cur_input;
  const char *split_c = ",";
  cur_input = strtok(input_list, split_c);
  while (cur_input) {
    input_data_list.emplace_back(cur_input);
    cur_input = strtok(nullptr, split_c);
  }
  delete[] input_list;
}

void BenchmarkFlags::InitResizeDimsList() {
  std::string content;
  content = this->resizeDimsIn;
  std::vector<int64_t> shape;
  auto shapeStrs = StringSplit(content, std::string(DELIM_COLON));
  for (const auto &shapeStr : shapeStrs) {
    shape.clear();
    auto dimStrs = StringSplit(shapeStr, std::string(DELIM_COMMA));
    std::cout << "Resize Dims: ";
    for (const auto &dimStr : dimStrs) {
      std::cout << dimStr << " ";
      shape.emplace_back(static_cast<int64_t>(std::stoi(dimStr)));
    }
    std::cout << std::endl;
    this->resizeDims.emplace_back(shape);
  }
}

int Benchmark::Init() {
  if (this->_flags == nullptr) {
    return 1;
  }
  MS_LOG(INFO) << "ModelPath = " << this->_flags->modelPath;
  MS_LOG(INFO) << "InDataPath = " << this->_flags->inDataPath;
  MS_LOG(INFO) << "InDataType = " << this->_flags->inDataTypeIn;
  MS_LOG(INFO) << "LoopCount = " << this->_flags->loopCount;
  MS_LOG(INFO) << "DeviceType = " << this->_flags->device;
  MS_LOG(INFO) << "AccuracyThreshold = " << this->_flags->accuracyThreshold;
  MS_LOG(INFO) << "WarmUpLoopCount = " << this->_flags->warmUpLoopCount;
  MS_LOG(INFO) << "NumThreads = " << this->_flags->numThreads;
  MS_LOG(INFO) << "calibDataPath = " << this->_flags->calibDataPath;
  if (this->_flags->cpuBindMode == -1) {
    MS_LOG(INFO) << "cpuBindMode = MID_CPU";
  } else if (this->_flags->cpuBindMode == 1) {
    MS_LOG(INFO) << "cpuBindMode = HIGHER_CPU";
  } else {
    MS_LOG(INFO) << "cpuBindMode = NO_BIND";
  }

  this->_flags->inDataType = this->_flags->inDataTypeIn == "img" ? kImage : kBinary;

  if (_flags->modelPath.empty()) {
    MS_LOG(ERROR) << "modelPath is required";
    return 1;
  }
  _flags->InitInputDataList();
  _flags->InitResizeDimsList();
  if (!_flags->resizeDims.empty() && _flags->resizeDims.size() != _flags->input_data_list.size()) {
    MS_LOG(ERROR) << "Size of input resizeDims should be equal to size of input inDataPath";
    return 1;
  }

  return 0;
}

int RunBenchmark(int argc, const char **argv) {
  BenchmarkFlags flags;
  Option<std::string> err = flags.ParseFlags(argc, argv);

  if (err.IsSome()) {
    std::cerr << err.Get() << std::endl;
    std::cerr << flags.Usage() << std::endl;
    return -1;
  }

  if (flags.help) {
    std::cerr << flags.Usage() << std::endl;
    return 0;
  }

  Benchmark mBenchmark(&flags);
  auto status = mBenchmark.Init();
  if (status != 0) {
    MS_LOG(ERROR) << "Benchmark init Error : " << status;
    return 1;
  }

  if (flags.device == "NPU") {
    status = mBenchmark.RunBenchmark("NPU");
  } else {
    status = mBenchmark.RunBenchmark("CPU");
  }

  if (status != 0) {
    MS_LOG(ERROR) << "Run Benchmark Error : " << status;
    return 1;
  }

  MS_LOG(INFO) << "end of benchmark";
  return 0;
}
}  // namespace lite
}  // namespace mindspore

