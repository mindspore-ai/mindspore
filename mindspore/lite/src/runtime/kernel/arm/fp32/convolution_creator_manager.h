/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_CREATOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_CREATOR_H_

#include <utility>
#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/conv_parameter.h"

namespace mindspore::lite {
using Category = lite::Tensor::Category;
using Format = mindspore::schema::Format;
struct TensorInfo {
  std::vector<int> shape_;
  Format format_;
  TypeId data_type_;
  Category tensor_type_;
  bool is_in_;
};

inline void CopyTensorQuantParam(lite::Tensor *dst, lite::Tensor *src) {
  for (size_t i = 0; i < src->quant_params().size(); i++) {
    dst->AddQuantParam(src->quant_params().at(i));
  }
}

inline ConvParameter *CreateNewConvParameter(ConvParameter *parameter) {
  auto conv_parameter = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (conv_parameter == nullptr) {
    MS_LOG(ERROR) << "Malloc new conv parameter failed.";
    return nullptr;
  }
  memcpy(conv_parameter, parameter, sizeof(ConvParameter));
  return conv_parameter;
}

inline void FreeMemory(ConvParameter *conv_param, const std::vector<lite::Tensor *> &new_inputs,
                       const std::vector<lite::Tensor *> &new_outputs) {
  if (conv_param != nullptr) {
    free(conv_param);
  }
  for (auto &in_tensor : new_inputs) {
    delete in_tensor;
  }
  for (auto &out_tensor : new_outputs) {
    delete out_tensor;
  }
}

lite::Tensor *CreateVarTensor(const TensorInfo &tensor_info, bool inferred);

lite::Tensor *CreateConstTensor(lite::Tensor *tensor, const std::vector<int> &shape, int index);

kernel::LiteKernel *CpuConvInt8KernelSelect(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                            const InnerContext *ctx);

kernel::LiteKernel *DispatchConvDw(const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                   const InnerContext *ctx);

kernel::LiteKernel *DispatchGroupConv(const std::vector<lite::Tensor *> &inputs,
                                      const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                      const InnerContext *ctx);

class GroupConvCreator {
 public:
  GroupConvCreator(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs, OpParameter *op_parameter,
                   const InnerContext *ctx, bool is_quant)
      : origin_inputs_(std::move(inputs)),
        origin_outputs_(std::move(outputs)),
        context_(ctx),
        infered_(op_parameter->infer_flag_),
        is_quant_(is_quant) {
    conv_param_ = reinterpret_cast<ConvParameter *>(op_parameter);
  }

  ~GroupConvCreator() = default;

 public:
  void SetShapeOfTensors();
  void set_input_shape(const std::vector<int> &shape) { input_shape_ = shape; }
  void set_output_shape(const std::vector<int> &shape) { output_shape_ = shape; }
  void set_filter_shape(const std::vector<int> &shape) { filter_shape_ = shape; }
  void set_bias_shape(const std::vector<int> &shape) { bias_shape_ = shape; }

  std::vector<kernel::LiteKernel *> get_group_conv() { return group_convs_; }
  int CreatGroupConv();

 protected:
  void FreeSubConv() {
    for (auto &sub_conv : group_convs_) {
      delete sub_conv;
    }
  }

  bool CheckIfValidPoint(void *ptr) {
    if (ptr == nullptr) {
      MS_LOG(ERROR) << "pointer is nullptr.";
      FreeSubConv();
      return false;
    }
    return true;
  }
  int NewInputTensor(std::vector<lite::Tensor *> *tensors) {
    auto in_tensor = CreateVarTensor(
      {input_shape_, Format::Format_NHWC, origin_inputs_.at(0)->data_type(), Category::VAR, true}, infered_);
    if (!CheckIfValidPoint(in_tensor)) {
      return RET_ERROR;
    }
    tensors->emplace_back(in_tensor);
    return RET_OK;
  }

  int NewConstTensor(std::vector<lite::Tensor *> *tensors, int group_id) {
    std::vector<std::pair<int, std::vector<int>>> const_tensor_list{std::make_pair(kWeightIndex, filter_shape_)};
    if (origin_inputs_.size() == 3) {
      const_tensor_list.emplace_back(std::make_pair(kBiasIndex, bias_shape_));
    }
    for (auto &info : const_tensor_list) {
      auto const_tensor = CreateConstTensor(origin_inputs_.at(info.first), info.second, group_id);
      if (!CheckIfValidPoint(const_tensor)) {
        return RET_ERROR;
      }
      tensors->emplace_back(const_tensor);
    }
    return RET_OK;
  }

  int NewOutputTensor(std::vector<lite::Tensor *> *tensors, lite::Tensor *output) {
    auto out_tensor =
      CreateVarTensor({output_shape_, output->format(), output->data_type(), output->category(), false}, infered_);
    if (!CheckIfValidPoint(out_tensor)) {
      return RET_ERROR;
    }
    if (is_quant_) {
      CopyTensorQuantParam(out_tensor, output);
    }
    tensors->emplace_back(out_tensor);
    return RET_OK;
  }

  void CopyQuantParam(std::vector<lite::Tensor *> *tensors) {
    for (size_t j = 0; j < origin_inputs_.size(); ++j) {
      CopyTensorQuantParam(tensors->at(j), origin_inputs_.at(j));
    }
  }

 private:
  std::vector<lite::Tensor *> origin_inputs_;
  std::vector<lite::Tensor *> origin_outputs_;
  std::vector<kernel::LiteKernel *> group_convs_;
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
  std::vector<int> filter_shape_;
  std::vector<int> bias_shape_;
  const InnerContext *context_;
  ConvParameter *conv_param_;
  bool infered_;
  bool is_quant_;
};

}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_CREATOR_H_
