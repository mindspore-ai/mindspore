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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_GROUP_CONVOLUTION_CREATOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_GROUP_CONVOLUTION_CREATOR_H_

#include <utility>
#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/conv_parameter.h"
#include "src/litert/tensor_category.h"

namespace mindspore::kernel {
struct TensorInfo {
  std::vector<int> shape_;
  mindspore::Format format_;
  TypeId data_type_;
  lite::Category tensor_type_;
  bool is_in_;
};

class GroupConvCreator {
 public:
  GroupConvCreator(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs, OpParameter *op_parameter,
                   bool is_quant, TypeId data_type)
      : origin_inputs_(std::move(inputs)),
        origin_outputs_(std::move(outputs)),
        is_quant_(is_quant),
        data_type_(data_type) {
    auto shape = origin_outputs_.front()->shape();
    infered_ = std::find(shape.begin(), shape.end(), -1) == shape.end();
    conv_param_ = reinterpret_cast<ConvParameter *>(op_parameter);
  }

  ~GroupConvCreator() = default;

  int SetShapeOfTensors();
  int CreateConvs(std::vector<kernel::LiteKernel *> *group_convs);
  void CopyQuantParam(const std::vector<lite::Tensor *> *tensors);
  int GetSingleConvParam(ConvParameter *conv_param, std::vector<lite::Tensor *> *new_inputs,
                         std::vector<lite::Tensor *> *new_outputs, int group_id);

 protected:
  void set_input_shape(const std::vector<int> &shape) { input_shape_ = shape; }
  void set_output_shape(const std::vector<int> &shape) { output_shape_ = shape; }
  void set_filter_shape(const std::vector<int> &shape) { filter_shape_ = shape; }
  void set_bias_shape(const std::vector<int> &shape) { bias_shape_ = shape; }
  void FreeGroupConvs();
  int NewInputTensor(std::vector<lite::Tensor *> *tensors);
  int NewConstTensor(std::vector<lite::Tensor *> *tensors, int group_id);
  int NewOutputTensor(std::vector<lite::Tensor *> *tensors, const lite::Tensor *output) const;

 private:
  std::vector<lite::Tensor *> origin_inputs_;
  std::vector<lite::Tensor *> origin_outputs_;
  std::vector<kernel::LiteKernel *> group_convs_;
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
  std::vector<int> filter_shape_;
  std::vector<int> bias_shape_;
  ConvParameter *conv_param_ = nullptr;
  bool infered_ = false;
  bool is_quant_ = false;
  TypeId data_type_;
};

ConvParameter *CreateNewConvParameter(const ConvParameter *parameter);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_GROUP_CONVOLUTION_CREATOR_H_
