/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/tensorrt_context.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
TensorRTContext::~TensorRTContext() {
  if (network_ != nullptr) {
    network_->destroy();
    network_ = nullptr;
  }
  for (auto ptr : owner_memorys_) {
    free(ptr);
  }
}

bool TensorRTContext::Init() {
  network_ = runtime_->GetBuilder()->createNetworkV2(
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  if (network_ == nullptr) {
    MS_LOG(ERROR) << "New network init failed.";
    return false;
  }
  return true;
}

void TensorRTContext::SetRuntime(TensorRTRuntime *runtime) { runtime_ = runtime; }

nvinfer1::INetworkDefinition *TensorRTContext::network() { return network_; }

void TensorRTContext::RegisterLayer(nvinfer1::ILayer *layer, const std::string &basename) {
  if (layer == nullptr) {
    MS_LOG(ERROR) << "Register null layer!";
    return;
  }
  MS_LOG(DEBUG) << "ms_layer " << basename << " register";
  layer->setName((basename + "_" + std::to_string(counter_++)).c_str());
}

void TensorRTContext::RegisterTensor(ITensorHelper tensor, const std::string &basename) {
  std::string trt_name = basename + "_" + std::to_string(counter_++);
  tensor.trt_tensor_->setName(trt_name.c_str());
  MS_LOG(DEBUG) << "ms_tensor " << basename << " register to " << trt_name;
  ms_name2trt_tensor_[basename] = tensor;
}

void TensorRTContext::RegisterTensorWithSameName(ITensorHelper tensor, const std::string &basename) {
  std::string trt_name = basename;
  tensor.trt_tensor_->setName(trt_name.c_str());
  MS_LOG(DEBUG) << "ms_tensor " << basename << " register to " << trt_name;
  ms_name2trt_tensor_[basename] = tensor;
}

bool TensorRTContext::HasTensor(const std::string &name) const {
  return ms_name2trt_tensor_.find(name) != ms_name2trt_tensor_.end();
}

ITensorHelper TensorRTContext::MsName2Tensor(const std::string &ms_name) {
  if (ms_name2trt_tensor_.find(ms_name) != ms_name2trt_tensor_.end()) {
    return ms_name2trt_tensor_[ms_name];
  }
  MS_LOG(WARNING) << "Get Tensorrt tensor by ms_tensor: " << ms_name << " fail!";
  return {};
}

template <typename T>
nvinfer1::ITensor *TensorRTContext::ConvertTo0DTensor(T value) {
  void *ptr = malloc(sizeof(T));
  memcpy(ptr, reinterpret_cast<const void *>(&value), sizeof(T));
  owner_memorys_.push_back(ptr);

  nvinfer1::Weights weights{GetNvinferDataType<T>(), ptr, 1};
  nvinfer1::Dims dims{};
  nvinfer1::IConstantLayer *constant_tensor = network()->addConstant(dims, weights);
  if (constant_tensor == nullptr) {
    MS_LOG(ERROR) << "create constant_tensor failed.";
    return nullptr;
  }
  return constant_tensor->getOutput(0);
}

template <typename T>
nvinfer1::ITensor *TensorRTContext::ConvertTo1DTensor(T value) {
  return ConvertTo1DTensor(std::vector<T>{value});
}

template <typename T>
nvinfer1::ITensor *TensorRTContext::ConvertTo1DTensor(const std::vector<T> &values) {
  void *ptr = malloc(values.size() * sizeof(T));
  const T *begin = &values[0];
  memcpy(ptr, reinterpret_cast<const void *>(begin), values.size() * sizeof(T));
  owner_memorys_.push_back(ptr);

  nvinfer1::Weights weights{GetNvinferDataType<T>(), ptr, values.size()};
  nvinfer1::Dims dims{1, {values.size()}};
  nvinfer1::IConstantLayer *constant_tensor = network()->addConstant(dims, weights);
  if (constant_tensor == nullptr) {
    MS_LOG(ERROR) << "create constant_tensor failed.";
    return nullptr;
  }
  return constant_tensor->getOutput(0);
}

template nvinfer1::ITensor *TensorRTContext::ConvertTo0DTensor(int value);
template nvinfer1::ITensor *TensorRTContext::ConvertTo0DTensor(float value);
template nvinfer1::ITensor *TensorRTContext::ConvertTo1DTensor(int value);
template nvinfer1::ITensor *TensorRTContext::ConvertTo1DTensor(float value);
template nvinfer1::ITensor *TensorRTContext::ConvertTo1DTensor(const std::vector<int> &values);
template nvinfer1::ITensor *TensorRTContext::ConvertTo1DTensor(const std::vector<float> &values);
}  // namespace mindspore::lite
