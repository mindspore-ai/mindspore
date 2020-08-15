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

#include "c_ops/conv2d_grad_filter.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int Conv2DGradFilter::GetFormat() const { return this->primitive->value.AsConv2DGradFilter()->format; }
int Conv2DGradFilter::GetGroup() const { return this->primitive->value.AsConv2DGradFilter()->group; }
int Conv2DGradFilter::GetChannelIn() const { return this->primitive->value.AsConv2DGradFilter()->channelIn; }
int Conv2DGradFilter::GetChannelOut() const { return this->primitive->value.AsConv2DGradFilter()->channelOut; }
int Conv2DGradFilter::GetKernelW() const { return this->primitive->value.AsConv2DGradFilter()->kernelW; }
int Conv2DGradFilter::GetKernelH() const { return this->primitive->value.AsConv2DGradFilter()->kernelH; }
int Conv2DGradFilter::GetStrideW() const { return this->primitive->value.AsConv2DGradFilter()->strideW; }
int Conv2DGradFilter::GetStrideH() const { return this->primitive->value.AsConv2DGradFilter()->strideH; }
int Conv2DGradFilter::GetPadMode() const { return this->primitive->value.AsConv2DGradFilter()->padMode; }
int Conv2DGradFilter::GetPadUp() const { return this->primitive->value.AsConv2DGradFilter()->padUp; }
int Conv2DGradFilter::GetPadDown() const { return this->primitive->value.AsConv2DGradFilter()->padDown; }
int Conv2DGradFilter::GetPadLeft() const { return this->primitive->value.AsConv2DGradFilter()->padLeft; }
int Conv2DGradFilter::GetPadRight() const { return this->primitive->value.AsConv2DGradFilter()->padRight; }
int Conv2DGradFilter::GetDilateW() const { return this->primitive->value.AsConv2DGradFilter()->dilateW; }
int Conv2DGradFilter::GetDilateH() const { return this->primitive->value.AsConv2DGradFilter()->dilateH; }
bool Conv2DGradFilter::GetHasBias() const { return this->primitive->value.AsConv2DGradFilter()->hasBias; }
int Conv2DGradFilter::GetActivationType() const { return this->primitive->value.AsConv2DGradFilter()->activationType; }

void Conv2DGradFilter::SetFormat(int format) {
  this->primitive->value.AsConv2DGradFilter()->format = (schema::Format)format;
}
void Conv2DGradFilter::SetGroup(int group) { this->primitive->value.AsConv2DGradFilter()->group = group; }
void Conv2DGradFilter::SetChannelIn(int channel_in) {
  this->primitive->value.AsConv2DGradFilter()->channelIn = channel_in;
}
void Conv2DGradFilter::SetChannelOut(int channel_out) {
  this->primitive->value.AsConv2DGradFilter()->channelOut = channel_out;
}
void Conv2DGradFilter::SetKernelW(int kernel_w) { this->primitive->value.AsConv2DGradFilter()->kernelW = kernel_w; }
void Conv2DGradFilter::SetKernelH(int kernel_h) { this->primitive->value.AsConv2DGradFilter()->kernelH = kernel_h; }
void Conv2DGradFilter::SetStrideW(int stride_w) { this->primitive->value.AsConv2DGradFilter()->strideW = stride_w; }
void Conv2DGradFilter::SetStrideH(int stride_h) { this->primitive->value.AsConv2DGradFilter()->strideH = stride_h; }
void Conv2DGradFilter::SetPadMode(int pad_mode) {
  this->primitive->value.AsConv2DGradFilter()->padMode = (schema::PadMode)pad_mode;
}
void Conv2DGradFilter::SetPadUp(int pad_up) { this->primitive->value.AsConv2DGradFilter()->padUp = pad_up; }
void Conv2DGradFilter::SetPadDown(int pad_down) { this->primitive->value.AsConv2DGradFilter()->padDown = pad_down; }
void Conv2DGradFilter::SetPadLeft(int pad_left) { this->primitive->value.AsConv2DGradFilter()->padLeft = pad_left; }
void Conv2DGradFilter::SetPadRight(int pad_right) { this->primitive->value.AsConv2DGradFilter()->padRight = pad_right; }
void Conv2DGradFilter::SetDilateW(int dilate_w) { this->primitive->value.AsConv2DGradFilter()->dilateW = dilate_w; }
void Conv2DGradFilter::SetDilateH(int dilate_h) { this->primitive->value.AsConv2DGradFilter()->dilateH = dilate_h; }
void Conv2DGradFilter::SetHasBias(bool has_bias) { this->primitive->value.AsConv2DGradFilter()->hasBias = has_bias; }
void Conv2DGradFilter::SetActivationType(int activation_type) {
  this->primitive->value.AsConv2DGradFilter()->activationType = (schema::ActivationType)activation_type;
}

#else

int Conv2DGradFilter::GetFormat() const { return this->primitive->value_as_Conv2DGradFilter()->format(); }
int Conv2DGradFilter::GetGroup() const { return this->primitive->value_as_Conv2DGradFilter()->group(); }
int Conv2DGradFilter::GetChannelIn() const { return this->primitive->value_as_Conv2DGradFilter()->channelIn(); }
int Conv2DGradFilter::GetChannelOut() const { return this->primitive->value_as_Conv2DGradFilter()->channelOut(); }
int Conv2DGradFilter::GetKernelW() const { return this->primitive->value_as_Conv2DGradFilter()->kernelW(); }
int Conv2DGradFilter::GetKernelH() const { return this->primitive->value_as_Conv2DGradFilter()->kernelH(); }
int Conv2DGradFilter::GetStrideW() const { return this->primitive->value_as_Conv2DGradFilter()->strideW(); }
int Conv2DGradFilter::GetStrideH() const { return this->primitive->value_as_Conv2DGradFilter()->strideH(); }
int Conv2DGradFilter::GetPadMode() const { return this->primitive->value_as_Conv2DGradFilter()->padMode(); }
int Conv2DGradFilter::GetPadUp() const { return this->primitive->value_as_Conv2DGradFilter()->padUp(); }
int Conv2DGradFilter::GetPadDown() const { return this->primitive->value_as_Conv2DGradFilter()->padDown(); }
int Conv2DGradFilter::GetPadLeft() const { return this->primitive->value_as_Conv2DGradFilter()->padLeft(); }
int Conv2DGradFilter::GetPadRight() const { return this->primitive->value_as_Conv2DGradFilter()->padRight(); }
int Conv2DGradFilter::GetDilateW() const { return this->primitive->value_as_Conv2DGradFilter()->dilateW(); }
int Conv2DGradFilter::GetDilateH() const { return this->primitive->value_as_Conv2DGradFilter()->dilateH(); }
bool Conv2DGradFilter::GetHasBias() const { return this->primitive->value_as_Conv2DGradFilter()->hasBias(); }
int Conv2DGradFilter::GetActivationType() const {
  return this->primitive->value_as_Conv2DGradFilter()->activationType();
}

void Conv2DGradFilter::SetFormat(int format) {}
void Conv2DGradFilter::SetGroup(int group) {}
void Conv2DGradFilter::SetChannelIn(int channel_in) {}
void Conv2DGradFilter::SetChannelOut(int channel_out) {}
void Conv2DGradFilter::SetKernelW(int kernel_w) {}
void Conv2DGradFilter::SetKernelH(int kernel_h) {}
void Conv2DGradFilter::SetStrideW(int stride_w) {}
void Conv2DGradFilter::SetStrideH(int stride_h) {}
void Conv2DGradFilter::SetPadMode(int pad_mode) {}
void Conv2DGradFilter::SetPadUp(int pad_up) {}
void Conv2DGradFilter::SetPadDown(int pad_down) {}
void Conv2DGradFilter::SetPadLeft(int pad_left) {}
void Conv2DGradFilter::SetPadRight(int pad_right) {}
void Conv2DGradFilter::SetDilateW(int dilate_w) {}
void Conv2DGradFilter::SetDilateH(int dilate_h) {}
void Conv2DGradFilter::SetHasBias(bool has_bias) {}
void Conv2DGradFilter::SetActivationType(int activation_type) {}
#endif
}  // namespace mindspore
