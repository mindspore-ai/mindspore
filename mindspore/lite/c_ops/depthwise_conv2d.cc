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

#include "c_ops/depthwise_conv2d.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int DepthwiseConv2D::GetFormat() const { return this->primitive->value.AsDepthwiseConv2D()->format; }
int DepthwiseConv2D::GetChannelIn() const { return this->primitive->value.AsDepthwiseConv2D()->channelIn; }
int DepthwiseConv2D::GetChannelMultiplier() const {
  return this->primitive->value.AsDepthwiseConv2D()->channelMultiplier;
}
int DepthwiseConv2D::GetKernelW() const { return this->primitive->value.AsDepthwiseConv2D()->kernelW; }
int DepthwiseConv2D::GetKernelH() const { return this->primitive->value.AsDepthwiseConv2D()->kernelH; }
int DepthwiseConv2D::GetStrideW() const { return this->primitive->value.AsDepthwiseConv2D()->strideW; }
int DepthwiseConv2D::GetStrideH() const { return this->primitive->value.AsDepthwiseConv2D()->strideH; }
int DepthwiseConv2D::GetPadMode() const { return this->primitive->value.AsDepthwiseConv2D()->padMode; }
int DepthwiseConv2D::GetPadUp() const { return this->primitive->value.AsDepthwiseConv2D()->padUp; }
int DepthwiseConv2D::GetPadDown() const { return this->primitive->value.AsDepthwiseConv2D()->padDown; }
int DepthwiseConv2D::GetPadLeft() const { return this->primitive->value.AsDepthwiseConv2D()->padLeft; }
int DepthwiseConv2D::GetPadRight() const { return this->primitive->value.AsDepthwiseConv2D()->padRight; }
int DepthwiseConv2D::GetDilateW() const { return this->primitive->value.AsDepthwiseConv2D()->dilateW; }
int DepthwiseConv2D::GetDilateH() const { return this->primitive->value.AsDepthwiseConv2D()->dilateH; }
bool DepthwiseConv2D::GetHasBias() const { return this->primitive->value.AsDepthwiseConv2D()->hasBias; }
int DepthwiseConv2D::GetActivationType() const { return this->primitive->value.AsDepthwiseConv2D()->activationType; }

void DepthwiseConv2D::SetFormat(int format) {
  this->primitive->value.AsDepthwiseConv2D()->format = (schema::Format)format;
}
void DepthwiseConv2D::SetChannelIn(int channel_in) {
  this->primitive->value.AsDepthwiseConv2D()->channelIn = channel_in;
}
void DepthwiseConv2D::SetChannelMultiplier(int channel_multiplier) {
  this->primitive->value.AsDepthwiseConv2D()->channelMultiplier = channel_multiplier;
}
void DepthwiseConv2D::SetKernelW(int kernel_w) { this->primitive->value.AsDepthwiseConv2D()->kernelW = kernel_w; }
void DepthwiseConv2D::SetKernelH(int kernel_h) { this->primitive->value.AsDepthwiseConv2D()->kernelH = kernel_h; }
void DepthwiseConv2D::SetStrideW(int stride_w) { this->primitive->value.AsDepthwiseConv2D()->strideW = stride_w; }
void DepthwiseConv2D::SetStrideH(int stride_h) { this->primitive->value.AsDepthwiseConv2D()->strideH = stride_h; }
void DepthwiseConv2D::SetPadMode(int pad_mode) {
  this->primitive->value.AsDepthwiseConv2D()->padMode = (schema::PadMode)pad_mode;
}
void DepthwiseConv2D::SetPadUp(int pad_up) { this->primitive->value.AsDepthwiseConv2D()->padUp = pad_up; }
void DepthwiseConv2D::SetPadDown(int pad_down) { this->primitive->value.AsDepthwiseConv2D()->padDown = pad_down; }
void DepthwiseConv2D::SetPadLeft(int pad_left) { this->primitive->value.AsDepthwiseConv2D()->padLeft = pad_left; }
void DepthwiseConv2D::SetPadRight(int pad_right) { this->primitive->value.AsDepthwiseConv2D()->padRight = pad_right; }
void DepthwiseConv2D::SetDilateW(int dilate_w) { this->primitive->value.AsDepthwiseConv2D()->dilateW = dilate_w; }
void DepthwiseConv2D::SetDilateH(int dilate_h) { this->primitive->value.AsDepthwiseConv2D()->dilateH = dilate_h; }
void DepthwiseConv2D::SetHasBias(bool has_bias) { this->primitive->value.AsDepthwiseConv2D()->hasBias = has_bias; }
void DepthwiseConv2D::SetActivationType(int activation_type) {
  this->primitive->value.AsDepthwiseConv2D()->activationType = (schema::ActivationType)activation_type;
}

#else

int DepthwiseConv2D::GetFormat() const { return this->primitive->value_as_DepthwiseConv2D()->format(); }
int DepthwiseConv2D::GetChannelIn() const { return this->primitive->value_as_DepthwiseConv2D()->channelIn(); }
int DepthwiseConv2D::GetChannelMultiplier() const {
  return this->primitive->value_as_DepthwiseConv2D()->channelMultiplier();
}
int DepthwiseConv2D::GetKernelW() const { return this->primitive->value_as_DepthwiseConv2D()->kernelW(); }
int DepthwiseConv2D::GetKernelH() const { return this->primitive->value_as_DepthwiseConv2D()->kernelH(); }
int DepthwiseConv2D::GetStrideW() const { return this->primitive->value_as_DepthwiseConv2D()->strideW(); }
int DepthwiseConv2D::GetStrideH() const { return this->primitive->value_as_DepthwiseConv2D()->strideH(); }
int DepthwiseConv2D::GetPadMode() const { return this->primitive->value_as_DepthwiseConv2D()->padMode(); }
int DepthwiseConv2D::GetPadUp() const { return this->primitive->value_as_DepthwiseConv2D()->padUp(); }
int DepthwiseConv2D::GetPadDown() const { return this->primitive->value_as_DepthwiseConv2D()->padDown(); }
int DepthwiseConv2D::GetPadLeft() const { return this->primitive->value_as_DepthwiseConv2D()->padLeft(); }
int DepthwiseConv2D::GetPadRight() const { return this->primitive->value_as_DepthwiseConv2D()->padRight(); }
int DepthwiseConv2D::GetDilateW() const { return this->primitive->value_as_DepthwiseConv2D()->dilateW(); }
int DepthwiseConv2D::GetDilateH() const { return this->primitive->value_as_DepthwiseConv2D()->dilateH(); }
bool DepthwiseConv2D::GetHasBias() const { return this->primitive->value_as_DepthwiseConv2D()->hasBias(); }
int DepthwiseConv2D::GetActivationType() const { return this->primitive->value_as_DepthwiseConv2D()->activationType(); }

void DepthwiseConv2D::SetFormat(int format) {}
void DepthwiseConv2D::SetChannelIn(int channel_in) {}
void DepthwiseConv2D::SetChannelMultiplier(int channel_multiplier) {}
void DepthwiseConv2D::SetKernelW(int kernel_w) {}
void DepthwiseConv2D::SetKernelH(int kernel_h) {}
void DepthwiseConv2D::SetStrideW(int stride_w) {}
void DepthwiseConv2D::SetStrideH(int stride_h) {}
void DepthwiseConv2D::SetPadMode(int pad_mode) {}
void DepthwiseConv2D::SetPadUp(int pad_up) {}
void DepthwiseConv2D::SetPadDown(int pad_down) {}
void DepthwiseConv2D::SetPadLeft(int pad_left) {}
void DepthwiseConv2D::SetPadRight(int pad_right) {}
void DepthwiseConv2D::SetDilateW(int dilate_w) {}
void DepthwiseConv2D::SetDilateH(int dilate_h) {}
void DepthwiseConv2D::SetHasBias(bool has_bias) {}
void DepthwiseConv2D::SetActivationType(int activation_type) {}
#endif
int DepthwiseConv2D::InferShape(std::vector<lite::tensor::Tensor *> inputs_,
                                std::vector<lite::tensor::Tensor *> outputs_) {
  if (inputs_.size() != kDoubleNum && inputs_.size() != kMultiNum) {
    MS_LOG(ERROR) << "inputs number is invalid";
    return 1;
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "output number is invalid";
    return 1;
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

  pad_l_ = GetPadLeft();
  pad_u_ = GetPadUp();
  pad_d_ = GetPadDown();
  pad_r_ = GetPadRight();
  if (GetPadMode() == schema::PadMode_SAME) {
    output_h = std::ceil(static_cast<float>(input_h) / static_cast<float>(GetStrideH()));
    output_w = std::ceil(static_cast<float>(input_w) / static_cast<float>(GetStrideW()));
    auto pad_h_all = ((output_h - 1) * GetStrideH() + (GetKernelH() - 1) * GetDilateH() + 1 - input_h);
    auto pad_w_all = ((output_w - 1) * GetStrideW() + (GetKernelW() - 1) * GetDilateW() + 1 - input_w);
    pad_u_ = pad_h_all / 2;
    pad_d_ = pad_h_all - pad_u_;
    pad_l_ = pad_w_all / 2;
    pad_r_ = pad_w_all - pad_l_;
  } else {
    output_h = std::ceil((static_cast<float>(input_h) + pad_u_ + pad_d_ -
                          (static_cast<float>(GetKernelH()) - 1) * static_cast<float>(GetDilateH())) /
                         static_cast<float>(GetStrideH()));
    output_w = std::ceil((static_cast<float>(input_w) + pad_l_ + pad_r_ -
                          (static_cast<float>(GetKernelW()) - 1) * static_cast<float>(GetDilateW())) /
                         static_cast<float>(GetStrideW()));
  }
  std::vector<int> out_shape{input->shape()};
  out_shape.at(1) = output_h;
  out_shape.at(2) = output_w;
  if (GetChannelMultiplier() * input_channel != weight->shape()[0]) {
    MS_LOG(ERROR) << "Conv depthwise only support group equals output channel.";
    return 1;
  }
  out_shape.at(3) = weight->shape()[0] * weight->shape()[3];  // in_channel * out_channel

  output->set_shape(out_shape);
  output->SetFormat(input->GetFormat());
  output->set_data_type(input->data_type());
  return 0;
}
}  // namespace mindspore
