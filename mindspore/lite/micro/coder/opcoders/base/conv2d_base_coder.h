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
#include "micro/coder/opcoders/op_coder.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"
#include "nnacl/conv_parameter.h"
namespace mindspore::lite::micro {

using std::string;

class Conv2DBaseCoder : public OperatorCoder {
 public:
  Conv2DBaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~Conv2DBaseCoder() override {
    if (conv_quant_arg_ == nullptr) {
      return;
    }
    free(conv_quant_arg_->real_multiplier_);
    free(conv_quant_arg_->left_shift_);
    free(conv_quant_arg_->right_shift_);
    free(conv_quant_arg_->quant_multiplier_);
    free(conv_quant_arg_->out_act_min_);
    free(conv_quant_arg_->out_act_max_);
    free(conv_quant_arg_->input_quant_args_);
    free(conv_quant_arg_->filter_quant_args_);
    free(conv_quant_arg_->output_quant_args_);
  }

 protected:
  int Init();

  int SetQuantParam();

  int MallocQuantParam();

  int SetInputTensorQuantParam();

  int SetFilterTensorQuantParam();

  int SetOutputTensorQuantParam();

  int SetQuantMultiplier();

  int CheckResizeValid();

  int SetIfPerChannel();

  int CheckLayout(lite::Tensor *input_tensor);

  string LayoutTransformFp32(schema::Format src_format, schema::Format dst_format);

  string LayoutTransformInt8(schema::Format src_format, schema::Format dst_format);

  string LayoutTransform(TypeId data_type, schema::Format src_format, schema::Format dst_format);

  ConvParameter *conv_param_{nullptr};

  ConvQuantArg *conv_quant_arg_{nullptr};

  Tensor *filter_tensor_{nullptr};

  Tensor *bias_tensor_{nullptr};

  string convert_func_;
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_BASE_CONV2D_BASE_CODER_H_
