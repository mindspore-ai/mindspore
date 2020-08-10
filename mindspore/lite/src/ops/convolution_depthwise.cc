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
int DepthwiseConv2D::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
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

  auto conv_prim = this->primitive->value_as_DepthwiseConv2D();
  pad_l_ = conv_prim->padLeft();
  pad_u_ = conv_prim->padUp();
  pad_d_ = conv_prim->padDown();
  pad_r_ = conv_prim->padRight();
  if (conv_prim->padMode() == schema::PadMode_SAME) {
    output_h = std::ceil(static_cast<float>(input_h) / static_cast<float>(conv_prim->strideH()));
    output_w = std::ceil(static_cast<float>(input_w) / static_cast<float>(conv_prim->strideW()));
    auto pad_h_all =
      ((output_h - 1) * conv_prim->strideH() + (conv_prim->kernelH() - 1) * conv_prim->dilateH() + 1 - input_h);
    auto pad_w_all =
      ((output_w - 1) * conv_prim->strideW() + (conv_prim->kernelW() - 1) * conv_prim->dilateW() + 1 - input_w);
    pad_u_ = pad_h_all / 2;
    pad_d_ = pad_h_all - pad_u_;
    pad_l_ = pad_w_all / 2;
    pad_r_ = pad_w_all - pad_l_;
  } else {
    output_h =
      std::ceil((static_cast<float>(input_h) + pad_u_ + pad_d_ - (static_cast<float>(conv_prim->kernelH()) - 1) *
          static_cast<float>(conv_prim->dilateH())) / static_cast<float>(conv_prim->strideH()));
    output_w =
      std::ceil((static_cast<float>(input_w) + pad_l_ + pad_r_ - (static_cast<float>(conv_prim->kernelW()) - 1) *
          static_cast<float>(conv_prim->dilateW())) / static_cast<float>(conv_prim->strideW()));
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

