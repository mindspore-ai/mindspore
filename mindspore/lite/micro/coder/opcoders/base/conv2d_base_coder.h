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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_BASE_CONV2D_BASE_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_BASE_CONV2D_BASE_CODER_H_

#include <string>
#include <vector>
#include <utility>
#include <memory>
#include "coder/opcoders/op_coder.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"
#include "nnacl/conv_parameter.h"
namespace mindspore::lite::micro {

class Conv2DBaseCoder : public OperatorCoder {
 public:
  Conv2DBaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~Conv2DBaseCoder() override;

 protected:
  virtual int Init();

  int SetQuantParam();

  int MallocQuantParam();

  int SetInputTensorQuantParam();

  int SetFilterTensorQuantParam();

  int SetOutputTensorQuantParam();

  void SetRoundingAndMultipilerMode();

  int SetQuantMultiplier();

  int CheckResizeValid();

  int SetIfPerChannel();

  int CheckLayout(lite::Tensor *input_tensor);

  std::string LayoutTransformFp32(schema::Format src_format, schema::Format dst_format);

  std::string LayoutTransformInt8(schema::Format src_format, schema::Format dst_format);

  std::string LayoutTransform(TypeId data_type, schema::Format src_format, schema::Format dst_format);

 private:
  int MallocConvQuantParams(size_t input_arg_num, size_t filter_arg_num, size_t output_arg_num);
  void FreeConvQuantParams();

 protected:
  ConvParameter *conv_param_{nullptr};

  ConvQuantArg *conv_quant_arg_{nullptr};

  Tensor *filter_tensor_{nullptr};

  Tensor *bias_tensor_{nullptr};

  std::string convert_func_;
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_BASE_CONV2D_BASE_CODER_H_
