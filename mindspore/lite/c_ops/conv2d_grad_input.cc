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

#include "c_ops/conv2d_grad_input.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int Conv2DGradInput::GetFormat() const { return this->primitive->value.AsConv2DGradInput()->format; }
int Conv2DGradInput::GetGroup() const { return this->primitive->value.AsConv2DGradInput()->group; }
int Conv2DGradInput::GetChannelIn() const { return this->primitive->value.AsConv2DGradInput()->channelIn; }
int Conv2DGradInput::GetChannelOut() const { return this->primitive->value.AsConv2DGradInput()->channelOut; }
int Conv2DGradInput::GetKernelW() const { return this->primitive->value.AsConv2DGradInput()->kernelW; }
int Conv2DGradInput::GetKernelH() const { return this->primitive->value.AsConv2DGradInput()->kernelH; }
int Conv2DGradInput::GetStrideW() const { return this->primitive->value.AsConv2DGradInput()->strideW; }
int Conv2DGradInput::GetStrideH() const { return this->primitive->value.AsConv2DGradInput()->strideH; }
int Conv2DGradInput::GetPadMode() const { return this->primitive->value.AsConv2DGradInput()->padMode; }
int Conv2DGradInput::GetPadUp() const { return this->primitive->value.AsConv2DGradInput()->padUp; }
int Conv2DGradInput::GetPadDown() const { return this->primitive->value.AsConv2DGradInput()->padDown; }
int Conv2DGradInput::GetPadLeft() const { return this->primitive->value.AsConv2DGradInput()->padLeft; }
int Conv2DGradInput::GetPadRight() const { return this->primitive->value.AsConv2DGradInput()->padRight; }
int Conv2DGradInput::GetDilateW() const { return this->primitive->value.AsConv2DGradInput()->dilateW; }
int Conv2DGradInput::GetDilateH() const { return this->primitive->value.AsConv2DGradInput()->dilateH; }
bool Conv2DGradInput::GetHasBias() const { return this->primitive->value.AsConv2DGradInput()->hasBias; }
int Conv2DGradInput::GetActivationType() const { return this->primitive->value.AsConv2DGradInput()->activationType; }

void Conv2DGradInput::SetFormat(int format) {
  this->primitive->value.AsConv2DGradInput()->format = (schema::Format)format;
}
void Conv2DGradInput::SetGroup(int group) { this->primitive->value.AsConv2DGradInput()->group = group; }
void Conv2DGradInput::SetChannelIn(int channel_in) {
  this->primitive->value.AsConv2DGradInput()->channelIn = channel_in;
}
void Conv2DGradInput::SetChannelOut(int channel_out) {
  this->primitive->value.AsConv2DGradInput()->channelOut = channel_out;
}
void Conv2DGradInput::SetKernelW(int kernel_w) { this->primitive->value.AsConv2DGradInput()->kernelW = kernel_w; }
void Conv2DGradInput::SetKernelH(int kernel_h) { this->primitive->value.AsConv2DGradInput()->kernelH = kernel_h; }
void Conv2DGradInput::SetStrideW(int stride_w) { this->primitive->value.AsConv2DGradInput()->strideW = stride_w; }
void Conv2DGradInput::SetStrideH(int stride_h) { this->primitive->value.AsConv2DGradInput()->strideH = stride_h; }
void Conv2DGradInput::SetPadMode(int pad_mode) {
  this->primitive->value.AsConv2DGradInput()->padMode = (schema::PadMode)pad_mode;
}
void Conv2DGradInput::SetPadUp(int pad_up) { this->primitive->value.AsConv2DGradInput()->padUp = pad_up; }
void Conv2DGradInput::SetPadDown(int pad_down) { this->primitive->value.AsConv2DGradInput()->padDown = pad_down; }
void Conv2DGradInput::SetPadLeft(int pad_left) { this->primitive->value.AsConv2DGradInput()->padLeft = pad_left; }
void Conv2DGradInput::SetPadRight(int pad_right) { this->primitive->value.AsConv2DGradInput()->padRight = pad_right; }
void Conv2DGradInput::SetDilateW(int dilate_w) { this->primitive->value.AsConv2DGradInput()->dilateW = dilate_w; }
void Conv2DGradInput::SetDilateH(int dilate_h) { this->primitive->value.AsConv2DGradInput()->dilateH = dilate_h; }
void Conv2DGradInput::SetHasBias(bool has_bias) { this->primitive->value.AsConv2DGradInput()->hasBias = has_bias; }
void Conv2DGradInput::SetActivationType(int activation_type) {
  this->primitive->value.AsConv2DGradInput()->activationType = (schema::ActivationType)activation_type;
}

#else

int Conv2DGradInput::GetFormat() const { return this->primitive->value_as_Conv2DGradInput()->format(); }
int Conv2DGradInput::GetGroup() const { return this->primitive->value_as_Conv2DGradInput()->group(); }
int Conv2DGradInput::GetChannelIn() const { return this->primitive->value_as_Conv2DGradInput()->channelIn(); }
int Conv2DGradInput::GetChannelOut() const { return this->primitive->value_as_Conv2DGradInput()->channelOut(); }
int Conv2DGradInput::GetKernelW() const { return this->primitive->value_as_Conv2DGradInput()->kernelW(); }
int Conv2DGradInput::GetKernelH() const { return this->primitive->value_as_Conv2DGradInput()->kernelH(); }
int Conv2DGradInput::GetStrideW() const { return this->primitive->value_as_Conv2DGradInput()->strideW(); }
int Conv2DGradInput::GetStrideH() const { return this->primitive->value_as_Conv2DGradInput()->strideH(); }
int Conv2DGradInput::GetPadMode() const { return this->primitive->value_as_Conv2DGradInput()->padMode(); }
int Conv2DGradInput::GetPadUp() const { return this->primitive->value_as_Conv2DGradInput()->padUp(); }
int Conv2DGradInput::GetPadDown() const { return this->primitive->value_as_Conv2DGradInput()->padDown(); }
int Conv2DGradInput::GetPadLeft() const { return this->primitive->value_as_Conv2DGradInput()->padLeft(); }
int Conv2DGradInput::GetPadRight() const { return this->primitive->value_as_Conv2DGradInput()->padRight(); }
int Conv2DGradInput::GetDilateW() const { return this->primitive->value_as_Conv2DGradInput()->dilateW(); }
int Conv2DGradInput::GetDilateH() const { return this->primitive->value_as_Conv2DGradInput()->dilateH(); }
bool Conv2DGradInput::GetHasBias() const { return this->primitive->value_as_Conv2DGradInput()->hasBias(); }
int Conv2DGradInput::GetActivationType() const { return this->primitive->value_as_Conv2DGradInput()->activationType(); }

void Conv2DGradInput::SetFormat(int format) {}
void Conv2DGradInput::SetGroup(int group) {}
void Conv2DGradInput::SetChannelIn(int channel_in) {}
void Conv2DGradInput::SetChannelOut(int channel_out) {}
void Conv2DGradInput::SetKernelW(int kernel_w) {}
void Conv2DGradInput::SetKernelH(int kernel_h) {}
void Conv2DGradInput::SetStrideW(int stride_w) {}
void Conv2DGradInput::SetStrideH(int stride_h) {}
void Conv2DGradInput::SetPadMode(int pad_mode) {}
void Conv2DGradInput::SetPadUp(int pad_up) {}
void Conv2DGradInput::SetPadDown(int pad_down) {}
void Conv2DGradInput::SetPadLeft(int pad_left) {}
void Conv2DGradInput::SetPadRight(int pad_right) {}
void Conv2DGradInput::SetDilateW(int dilate_w) {}
void Conv2DGradInput::SetDilateH(int dilate_h) {}
void Conv2DGradInput::SetHasBias(bool has_bias) {}
void Conv2DGradInput::SetActivationType(int activation_type) {}
#endif
}  // namespace mindspore
