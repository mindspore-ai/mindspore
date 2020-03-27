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

#include "benchmark/benchmark.h"
#include <random>
#include <limits>
#include <algorithm>
#include <utility>
#include <memory>
#include "include/session.h"

namespace mindspore {
namespace predict {
STATUS Benchmark::GenerateRandomData(size_t size, void *data) {
  MS_ASSERT(data != nullptr);
  char *castedData = static_cast<char *>(data);
  for (size_t i = 0; i < size; i++) {
    castedData[i] = static_cast<char>(i);
  }
  return RET_OK;
}

STATUS Benchmark::GenerateInputData() {
  for (Tensor *tensor : msInputs) {
    MS_ASSERT(tensor != nullptr);
    auto ret = tensor->MallocData();
    if (ret != RET_OK) {
      MS_LOGE("MallocData for inTensor failed %d", ret);
      return ret;
    }
    MS_ASSERT(tensor->GetData() != nullptr);
    auto tensorByteSize = tensor->GetDataSize();
    auto status = GenerateRandomData(tensorByteSize, tensor->GetData());
    if (status != RET_OK) {
      MS_LOGE("GenerateRandomData for inTensor failed %d", status);
      return status;
    }
  }
  return RET_OK;
}

STATUS Benchmark::LoadInput() {
  size_t size = 0;
  char *graphBuf = ReadFile(_flags->modelPath.c_str(), &size);
  if (graphBuf == nullptr) {
    MS_LOGE("Load graph failed, path %s", _flags->modelPath.c_str());
    return RET_ERROR;
  }

  this->msInputs = session->GetInput();

  if (_flags->inDataPath.empty()) {
    auto status = GenerateInputData();
    if (status != RET_OK) {
      delete graphBuf;
      MS_LOGE("Generate input data error %d", status);
      return status;
    }
  } else {
    auto status = ReadInputFile();
    if (status != RET_OK) {
      delete graphBuf;
      MS_LOGE("ReadInputFile error, %d", status);
      return status;
    }
  }
  delete graphBuf;
  return RET_OK;
}

STATUS Benchmark::ReadInputFile() {
  MS_ASSERT(msInputs.size() <= 1);
  if (msInputs.empty()) {
    return RET_OK;
  }
  Tensor *inTensor = msInputs.at(0);
  MS_ASSERT(inTensor != nullptr);

  size_t size;
  char *binBuf = ReadFile(_flags->inDataPath.c_str(), &size);
  if (binBuf == nullptr) {
    return RET_ERROR;
  }
  auto tensorDataSize = inTensor->GetDataSize();
  if (size != tensorDataSize) {
    MS_LOGE("Input binary file size error, required: %zu, in fact: %zu", tensorDataSize, size);
    delete binBuf;
    return RET_ERROR;
  }
  inTensor->SetData(binBuf);
  binBuf = nullptr;

  return RET_OK;
}

// calibData is FP32
STATUS Benchmark::ReadCalibData() {
  const char *calibDataPath = _flags->calibDataPath.c_str();
  // read calib data
  std::ifstream inFile(calibDataPath);
  if (!inFile.good()) {
    MS_LOGE("file: %s is not exist", calibDataPath);
    return RET_PARAM_INVALID;
  }

  if (!inFile.is_open()) {
    MS_LOGE("file: %s open failed", calibDataPath);
    inFile.close();
    return RET_PARAM_INVALID;
  }

  std::string line;
  MS_LOGI("Start reading calibData file");
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

    std::unique_ptr<CheckTensor> checkTensor(new CheckTensor(dims, tensorData));
    this->calibData.insert(std::make_pair(tensorName, checkTensor.release()));
  }
  inFile.close();
  MS_LOGI("Finish reading calibData file");
  return RET_OK;
}

// tensorData need to be converter first
float Benchmark::CompareData(const std::string &nodeName, std::vector<int64_t> msShape, float *msTensorData) {
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
      MS_LOGE("%s", oss.str().c_str());
      return -1;
    }

    float meanBias = 0;
    std::ostringstream outputData;
    outputData << "Data of node " << nodeName << " : ";
    for (size_t j = 0; j < shapeSize; j++) {
      if (j < printNum) {
        outputData << msTensorData[j] << " ";
      }
      if (fabs(calibTensor->data.at(j)) > minFloatThr) {
        double bias = fabs(msTensorData[j] - calibTensor->data.at(j)) / fabs(calibTensor->data.at(j));
        meanBias += bias;
      }
    }
    meanBias /= shapeSize;
    MS_LOGI("%s", outputData.str().c_str());

    if (meanBias <= minFloatThr) {
      MS_LOGI("Mean bias of node %s : 0%%", nodeName.c_str());
    } else {
      MS_LOGI("Mean bias of node %s : %f%%", nodeName.c_str(), meanBias * percentage);
    }
    return meanBias;
  } else {
    MS_LOGI("%s is not in Source Model output", nodeName.c_str());
    return -1;
  }
}

STATUS Benchmark::CompareOutput(const std::map<NODE_ID, std::vector<Tensor *>> &msOutputs) {
  float totalBias = 0;
  int totalSize = 0;
  bool hasError = false;
  for (const auto &msOutput : msOutputs) {
    std::string nodeName = msOutput.first;
    auto tensors = msOutput.second;
    for (auto tensor : tensors) {
      MS_ASSERT(tensor->GetData() != nullptr);
      float bias = CompareData(nodeName, tensor->GetDims(), static_cast<float *>(tensor->GetData()));
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
      meanBias = totalBias / totalSize * percentage;
    } else {
      meanBias = 0;
    }

    MS_LOGI("Mean bias all node : %f%%", meanBias);

    if (meanBias > 1) {
      MS_LOGE("Mean bias of all nodes is too big: %f%%", meanBias);
      return RET_ERROR;
    } else {
      return RET_OK;
    }
  } else {
    MS_LOGE("Error in CompareData");
    return RET_ERROR;
  }
}

STATUS Benchmark::MarkPerformance() {
  MS_LOGI("Running warm up loops...");
  for (int i = 0; i < _flags->warmUpLoopCount; i++) {
    auto status = session->Run(msInputs);
    if (status != RET_OK) {
      MS_LOGE("Inference error %d", status);
      return status;
    }
  }

  MS_LOGI("Running benchmark loops...");
  uint64_t timeMin = maxTimeThr;
  uint64_t timeMax = 0;
  uint64_t timeAvg = 0;
  for (int i = 0; i < _flags->loopCount; i++) {
    uint64_t start = GetTimeUs();
    auto status = session->Run(msInputs);
    if (status != RET_OK) {
      MS_LOGE("Inference error %d", status);
      return status;
    }

    uint64_t end = GetTimeUs();
    uint64_t time = end - start;
    timeMin = std::min(timeMin, time);
    timeMax = std::max(timeMax, time);
    timeAvg += time;

    msOutputs = session->GetAllOutput();
    if (cleanData) {
      for (auto &msOutput : msOutputs) {
        for (auto &outputTensor : msOutput.second) {
          delete outputTensor;
        }
      }
      msOutputs.clear();
    }
  }
  if (_flags->loopCount > 0) {
    timeAvg /= _flags->loopCount;
    MS_LOGI("MinRunTime = %f ms, MaxRuntime = %f ms, AvgRunTime = %f ms", timeMin / US2MS, timeMax / US2MS,
            timeAvg / US2MS);
  }
  return RET_OK;
}

STATUS Benchmark::MarkAccuracy() {
  MS_LOGI("MarkAccuracy");

  auto status = session->Run(msInputs);
  if (status != RET_OK) {
    MS_LOGE("Inference error %d", status);
    return status;
  }
  msOutputs = session->GetAllOutput();

  ReadCalibData();
  status = CompareOutput(msOutputs);
  if (cleanData) {
    for (auto &msOutput : msOutputs) {
      for (auto &outputTensor : msOutput.second) {
        delete outputTensor;
      }
    }
    msOutputs.clear();
  }
  return status;
}

STATUS Benchmark::CleanData() {
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
  return RET_OK;
}

STATUS Benchmark::RunBenchmark() {
  // Load graph
  std::string comment = modelName;

  MS_LOGI("start reading model file");
  size_t size = 0;
  char *graphBuf = ReadFile(_flags->modelPath.c_str(), &size);
  if (graphBuf == nullptr) {
    MS_LOGE("Load graph failed while running %s", comment.c_str());
    return RET_ERROR;
  }

  uint64_t startPrepareTime = GetTimeUs();
  session = CreateSession(graphBuf, size, ctx);
  if (session == nullptr) {
    delete graphBuf;
    MS_LOGE("new session failed while running %s", comment.c_str());
    return RET_ERROR;
  }
  uint64_t endPrepareTime = GetTimeUs();
  MS_LOGI("PrepareTime = %f ms, ", (endPrepareTime - startPrepareTime) / US2MS);

  // Load input
  MS_LOGI("start generate input data");
  auto status = LoadInput();
  if (status != RET_OK) {
    delete graphBuf;
    MS_LOGE("Generate input data error");
    return status;
  }

  if (!_flags->calibDataPath.empty()) {
    status = MarkAccuracy();
    if (status != RET_OK) {
      delete graphBuf;
      MS_LOGE("Run MarkAccuracy error: %d", status);
      return status;
    }
  } else {
    status = MarkPerformance();
    if (status != RET_OK) {
      delete graphBuf;
      MS_LOGE("Run MarkPerformance error: %d", status);
      return status;
    }
  }

  CleanData();
  delete graphBuf;
  return RET_OK;
}

STATUS Benchmark::Init() {
  if (this->_flags == nullptr) {
    return RET_ERROR;
  }
  MS_LOGI("ModelPath = %s", this->_flags->modelPath.c_str());
  MS_LOGI("InDataPath = %s", this->_flags->inDataPath.c_str());
  MS_LOGI("TensorDataType = %s", this->_flags->tensorDataTypeIn.c_str());
  MS_LOGI("LoopCount = %d", this->_flags->loopCount);
  MS_LOGI("WarmUpLoopCount = %d", this->_flags->warmUpLoopCount);
  MS_LOGI("NumThreads = %d", this->_flags->numThreads);
  MS_LOGI("calibDataPath = %s", this->_flags->calibDataPath.c_str());

  this->_flags->inDataType = this->_flags->inDataTypeIn == "img" ? kImage : kBinary;
  if (this->_flags->tensorDataTypeIn == "float") {
    this->_flags->tensorDataType = DataType_DT_FLOAT;
  }

  if (_flags->modelPath.empty()) {
    MS_LOGE("modelPath is required");
    return RET_ERROR;
  }

  modelName = _flags->modelPath.substr(_flags->modelPath.find_last_of("/") + 1);

  return RET_OK;
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
  if (status != RET_OK) {
    MS_LOGE("Benchmark init Error : %d", status);
    return 1;
  }

  status = mBenchmark.RunBenchmark();
  if (status != RET_OK) {
    MS_LOGE("Run Benchmark Error : %d", status);
    return 1;
  }

  MS_LOGI("end of benchmark");
  return 0;
}
}  // namespace predict
}  // namespace mindspore
