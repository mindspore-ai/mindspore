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

#include "mindspore/lite/c_ops/conv2d.h"
namespace mindspore {
int Conv2D::PadUp() const { return this->pad_u_; }
int Conv2D::PadDown() const { return this->pad_d_; }
int Conv2D::PadLeft() const { return this->pad_l_; }
int Conv2D::PadRight() const { return this->pad_r_; }
#ifdef PRIMITIVE_WRITEABLE
int Conv2D::GetFormat() const { return this->primitive->value.AsConv2D()->format; }
int Conv2D::GetGroup() const { return this->primitive->value.AsConv2D()->group; }
int Conv2D::GetChannelIn() const { return this->primitive->value.AsConv2D()->channelIn; }
int Conv2D::GetChannelOut() const { return this->primitive->value.AsConv2D()->channelOut; }
int Conv2D::GetKernelW() const { return this->primitive->value.AsConv2D()->kernelW; }
int Conv2D::GetKernelH() const { return this->primitive->value.AsConv2D()->kernelH; }
int Conv2D::GetStrideW() const { return this->primitive->value.AsConv2D()->strideW; }
int Conv2D::GetStrideH() const { return this->primitive->value.AsConv2D()->strideH; }
int Conv2D::GetPadMode() const { return this->primitive->value.AsConv2D()->padMode; }
int Conv2D::GetPadUp() const { return this->primitive->value.AsConv2D()->padUp; }
int Conv2D::GetPadDown() const { return this->primitive->value.AsConv2D()->padDown; }
int Conv2D::GetPadLeft() const { return this->primitive->value.AsConv2D()->padLeft; }
int Conv2D::GetPadRight() const { return this->primitive->value.AsConv2D()->padRight; }
int Conv2D::GetDilateW() const { return this->primitive->value.AsConv2D()->dilateW; }
int Conv2D::GetDilateH() const { return this->primitive->value.AsConv2D()->dilateH; }
bool Conv2D::GetHasBias() const { return this->primitive->value.AsConv2D()->hasBias; }
int Conv2D::GetActivationType() const { return this->primitive->value.AsConv2D()->activationType; }

void Conv2D::SetFormat(int format) { this->primitive->value.AsConv2D()->format = (schema::Format)format; }
void Conv2D::SetGroup(int group) { this->primitive->value.AsConv2D()->group = group; }
void Conv2D::SetChannelIn(int channel_in) { this->primitive->value.AsConv2D()->channelIn = channel_in; }
void Conv2D::SetChannelOut(int channel_out) { this->primitive->value.AsConv2D()->channelOut = channel_out; }
void Conv2D::SetKernelW(int kernel_w) { this->primitive->value.AsConv2D()->kernelW = kernel_w; }
void Conv2D::SetKernelH(int kernel_h) { this->primitive->value.AsConv2D()->kernelH = kernel_h; }
void Conv2D::SetStrideW(int stride_w) { this->primitive->value.AsConv2D()->strideW = stride_w; }
void Conv2D::SetStrideH(int stride_h) { this->primitive->value.AsConv2D()->strideH = stride_h; }
void Conv2D::SetPadMode(int pad_mode) { this->primitive->value.AsConv2D()->padMode = (schema::PadMode)pad_mode; }
void Conv2D::SetPadUp(int pad_up) { this->primitive->value.AsConv2D()->padUp = pad_up; }
void Conv2D::SetPadDown(int pad_down) { this->primitive->value.AsConv2D()->padDown = pad_down; }
void Conv2D::SetPadLeft(int pad_left) { this->primitive->value.AsConv2D()->padLeft = pad_left; }
void Conv2D::SetPadRight(int pad_right) { this->primitive->value.AsConv2D()->padRight = pad_right; }
void Conv2D::SetDilateW(int dilate_w) { this->primitive->value.AsConv2D()->dilateW = dilate_w; }
void Conv2D::SetDilateH(int dilate_h) { this->primitive->value.AsConv2D()->dilateH = dilate_h; }
void Conv2D::SetHasBias(bool has_bias) { this->primitive->value.AsConv2D()->hasBias = has_bias; }
void Conv2D::SetActivationType(int activation_type) {
  this->primitive->value.AsConv2D()->activationType = (schema::ActivationType)activation_type;
}

#else

int Conv2D::GetFormat() const { return this->primitive->value_as_Conv2D()->format(); }
int Conv2D::GetGroup() const { return this->primitive->value_as_Conv2D()->group(); }
int Conv2D::GetChannelIn() const { return this->primitive->value_as_Conv2D()->channelIn(); }
int Conv2D::GetChannelOut() const { return this->primitive->value_as_Conv2D()->channelOut(); }
int Conv2D::GetKernelW() const { return this->primitive->value_as_Conv2D()->kernelW(); }
int Conv2D::GetKernelH() const { return this->primitive->value_as_Conv2D()->kernelH(); }
int Conv2D::GetStrideW() const { return this->primitive->value_as_Conv2D()->strideW(); }
int Conv2D::GetStrideH() const { return this->primitive->value_as_Conv2D()->strideH(); }
int Conv2D::GetPadMode() const { return this->primitive->value_as_Conv2D()->padMode(); }
int Conv2D::GetPadUp() const { return this->primitive->value_as_Conv2D()->padUp(); }
int Conv2D::GetPadDown() const { return this->primitive->value_as_Conv2D()->padDown(); }
int Conv2D::GetPadLeft() const { return this->primitive->value_as_Conv2D()->padLeft(); }
int Conv2D::GetPadRight() const { return this->primitive->value_as_Conv2D()->padRight(); }
int Conv2D::GetDilateW() const { return this->primitive->value_as_Conv2D()->dilateW(); }
int Conv2D::GetDilateH() const { return this->primitive->value_as_Conv2D()->dilateH(); }
bool Conv2D::GetHasBias() const { return this->primitive->value_as_Conv2D()->hasBias(); }
int Conv2D::GetActivationType() const { return this->primitive->value_as_Conv2D()->activationType(); }

void Conv2D::SetFormat(int format) {}
void Conv2D::SetGroup(int group) {}
void Conv2D::SetChannelIn(int channel_in) {}
void Conv2D::SetChannelOut(int channel_out) {}
void Conv2D::SetKernelW(int kernel_w) {}
void Conv2D::SetKernelH(int kernel_h) {}
void Conv2D::SetStrideW(int stride_w) {}
void Conv2D::SetStrideH(int stride_h) {}
void Conv2D::SetPadMode(int pad_mode) {}
void Conv2D::SetPadUp(int pad_up) {}
void Conv2D::SetPadDown(int pad_down) {}
void Conv2D::SetPadLeft(int pad_left) {}
void Conv2D::SetPadRight(int pad_right) {}
void Conv2D::SetDilateW(int dilate_w) {}
void Conv2D::SetDilateH(int dilate_h) {}
void Conv2D::SetHasBias(bool has_bias) {}
void Conv2D::SetActivationType(int activation_type) {}
#endif
void Conv2D::ConvInferShape(int input_h, int input_w, int *output_h, int *output_w) {
  MS_ASSERT(this->primitive != nullptr);
  int kernel_w = GetKernelW();
  int kernel_h = GetKernelH();
  int stride_w = GetStrideW();
  int stride_h = GetStrideH();
  int dilate_w = GetDilateW();
  int dilate_h = GetDilateH();
  pad_l_ = GetPadLeft();
  pad_u_ = GetPadUp();
  pad_d_ = GetPadDown();
  pad_r_ = GetPadRight();

  if (GetPadMode() == schema::PadMode_SAME) {
    *output_w = std::ceil(static_cast<float>(input_w) / static_cast<float>(stride_w));
    *output_h = std::ceil(static_cast<float>(input_h) / static_cast<float>(stride_h));
    auto pad_h_all = ((*output_h - 1) * stride_h + (kernel_h - 1) * dilate_h + 1 - input_h);
    auto pad_w_all = ((*output_w - 1) * stride_w + (kernel_w - 1) * dilate_w + 1 - input_w);
    pad_u_ = pad_h_all / 2;
    pad_d_ = pad_h_all - pad_u_;
    pad_l_ = pad_w_all / 2;
    pad_r_ = pad_w_all - pad_l_;
  } else {
    *output_w = std::ceil((static_cast<float>(input_w) + pad_l_ + pad_r_ -
                           (static_cast<float>(kernel_w) - 1) * static_cast<float>(dilate_w)) /
                          static_cast<float>(stride_w));
    *output_h = std::ceil((static_cast<float>(input_h) + pad_u_ + pad_d_ -
                           (static_cast<float>(kernel_h) - 1) * static_cast<float>(dilate_h)) /
                          static_cast<float>(stride_h));
  }
}

int Conv2D::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  if (inputs_.size() != 2 && inputs_.size() != 3) {
    MS_LOG(ERROR) << "Add should has two or three inputs";
    return 1;
  }
  if (outputs_.size() != 1) {
    MS_LOG(ERROR) << "Add should has one outputs";
    return 1;
  }
  auto *input_tensor = inputs_.front();
  auto *weight_tensor = inputs_.at(1);
  auto *out_tensor = outputs_.front();
  MS_ASSERT(input_tensor != nullptr);
  MS_ASSERT(out_tensor != nullptr);

  auto in_shape = input_tensor->shape();
  int input_h = in_shape.at(1);
  int input_w = in_shape.at(2);
  int output_w = 0, output_h = 0;

  this->ConvInferShape(input_h, input_w, &output_h, &output_w);

  std::vector<int> out_shape{input_tensor->shape()};
  out_shape.at(1) = output_h;
  out_shape.at(2) = output_w;
  out_shape.at(3) = weight_tensor->shape()[0];
  out_tensor->set_shape(out_shape);
  out_tensor->SetFormat(input_tensor->GetFormat());
  out_tensor->set_data_type(input_tensor->data_type());
  return 0;
}
}  // namespace mindspore
