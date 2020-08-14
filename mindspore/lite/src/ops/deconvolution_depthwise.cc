/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/ops.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
int DeconvDepthwiseConv2D::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  if (inputs_.size() != kDoubleNum && inputs_.size() != kMultiNum) {
    MS_LOG(ERROR) << "inputs number is invalid";
    return RET_INPUT_TENSOR_ERROR;
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "output number is invalid";
    return RET_INPUT_TENSOR_ERROR;
  }
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto weight = inputs_.at(1);
  MS_ASSERT(weight != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  auto in_shape = input->shape();
  int input_h = in_shape.at(1);
  int input_w = in_shape.at(2);
  int input_channel = in_shape.at(3);
  int output_w = 0, output_h = 0;

  auto conv_prim = this->primitive->value_as_DeDepthwiseConv2D();
  pad_l_ = conv_prim->padLeft();
  pad_u_ = conv_prim->padUp();
  pad_d_ = conv_prim->padDown();
  pad_r_ = conv_prim->padRight();
  output_h = conv_prim->strideH() * (input_h - 1) + conv_prim->kernelH() - pad_u_ - pad_d_;
  output_w = conv_prim->strideW() * (input_w - 1) + conv_prim->kernelW() - pad_l_ - pad_r_;
  if ((output_h + conv_prim->padUp() + conv_prim->padDown() - conv_prim->kernelH()) % conv_prim->strideH() != 0) {
    output_h += (output_h + conv_prim->padLeft() + conv_prim->padRight() - conv_prim->kernelH()) % conv_prim->strideH();
  }
  if ((output_w + conv_prim->padLeft() + conv_prim->padRight() - conv_prim->kernelW()) % conv_prim->strideW() != 0) {
    output_w += (output_w + conv_prim->padLeft() + conv_prim->padRight() - conv_prim->kernelW()) % conv_prim->strideW();
  }
  std::vector<int> out_shape{input->shape()};
  out_shape.at(1) = output_h;
  out_shape.at(2) = output_w;
  if (conv_prim->channelMultiplier() * input_channel != weight->shape()[0]) {
    MS_LOG(ERROR) << "Conv depthwise only support group equals output channel.";
    return RET_ERROR;
  }
  out_shape.at(3) = weight->shape()[0] * weight->shape()[3];  // in_channel * out_channel

  output->set_shape(out_shape);
  output->SetFormat(input->GetFormat());
  output->set_data_type(input->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite

