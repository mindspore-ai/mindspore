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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTITIMIZER_TRT_PASS_LAYER_INPUT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTITIMIZER_TRT_PASS_LAYER_INPUT_H_

#include <vector>
#include <NvInfer.h>

namespace mindspore::opt {
// Tensor-RT layer inputs include weight or tensor.
// Tensor: Anf-graph inputs or feature map which values change during inference.
// Weight: Anf-graph inputs or value node which remain unchanged during inference.
class LayerInput {
 public:
  LayerInput() : type_(InputType::kUnknown), weight_(), tensor_(nullptr) {}
  explicit LayerInput(nvinfer1::Weights &w, const std::vector<int64_t> &s)
      : type_(InputType::kWeight), weight_(w), tensor_(nullptr), shape_(s) {}
  explicit LayerInput(nvinfer1::ITensor *t, const std::vector<int64_t> &s)
      : type_(InputType::kTensor), weight_(), tensor_(t), shape_(s) {}

  bool IsTensor() const { return type_ == InputType::kTensor; }
  bool IsWeight() const { return type_ == InputType::kWeight; }

  const nvinfer1::Weights *weight() {
    if (!IsWeight()) {
      MS_LOG(WARNING) << "weight not initialized.";
      return nullptr;
    }
    return &weight_;
  }

  nvinfer1::ITensor *tensor() const {
    if (!IsTensor()) {
      MS_LOG(WARNING) << "tensor not initialized.";
      return nullptr;
    }
    return tensor_;
  }

  const std::vector<int64_t> &shape() const { return shape_; }

 private:
  enum class InputType : char { kUnknown = 0, kTensor, kWeight };
  InputType type_;
  // Keep the copy rather than point cause Weights created as a local variable.
  nvinfer1::Weights weight_;
  // Keep the point as ITensor created/held by nvinfer1::INetworkDefinition.
  nvinfer1::ITensor *tensor_;
  // Keep the shape of tensor or weight.
  std::vector<int64_t> shape_;
};
}  // namespace mindspore::opt

#endif  // MINDSPORE_CCSRC_BACKEND_OPTITIMIZER_TRT_PASS_LAYER_INPUT_H_
