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

#include "src/ops/dedepthwise_conv2d.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int DeDepthwiseConv2D::GetFormat() const { return this->primitive_->value.AsDeDepthwiseConv2D()->format; }
int DeDepthwiseConv2D::GetChannelIn() const { return this->primitive_->value.AsDeDepthwiseConv2D()->channelIn; }
int DeDepthwiseConv2D::GetChannelMultiplier() const {
  return this->primitive_->value.AsDeDepthwiseConv2D()->channelMultiplier;
}
int DeDepthwiseConv2D::GetKernelW() const { return this->primitive_->value.AsDeDepthwiseConv2D()->kernelW; }
int DeDepthwiseConv2D::GetKernelH() const { return this->primitive_->value.AsDeDepthwiseConv2D()->kernelH; }
int DeDepthwiseConv2D::GetStrideW() const { return this->primitive_->value.AsDeDepthwiseConv2D()->strideW; }
int DeDepthwiseConv2D::GetStrideH() const { return this->primitive_->value.AsDeDepthwiseConv2D()->strideH; }
int DeDepthwiseConv2D::GetPadMode() const { return this->primitive_->value.AsDeDepthwiseConv2D()->padMode; }
int DeDepthwiseConv2D::GetPadUp() const { return this->primitive_->value.AsDeDepthwiseConv2D()->padUp; }
int DeDepthwiseConv2D::GetPadDown() const { return this->primitive_->value.AsDeDepthwiseConv2D()->padDown; }
int DeDepthwiseConv2D::GetPadLeft() const { return this->primitive_->value.AsDeDepthwiseConv2D()->padLeft; }
int DeDepthwiseConv2D::GetPadRight() const { return this->primitive_->value.AsDeDepthwiseConv2D()->padRight; }
int DeDepthwiseConv2D::GetDilateW() const { return this->primitive_->value.AsDeDepthwiseConv2D()->dilateW; }
int DeDepthwiseConv2D::GetDilateH() const { return this->primitive_->value.AsDeDepthwiseConv2D()->dilateH; }
int DeDepthwiseConv2D::GetActivationType() const {
  return this->primitive_->value.AsDeDepthwiseConv2D()->activationType;
}

void DeDepthwiseConv2D::SetFormat(int format) {
  this->primitive_->value.AsDeDepthwiseConv2D()->format = static_cast<schema::Format>(format);
}
void DeDepthwiseConv2D::SetChannelIn(int channel_in) {
  this->primitive_->value.AsDeDepthwiseConv2D()->channelIn = channel_in;
}
void DeDepthwiseConv2D::SetChannelMultiplier(int channel_multiplier) {
  this->primitive_->value.AsDeDepthwiseConv2D()->channelMultiplier = channel_multiplier;
}
void DeDepthwiseConv2D::SetKernelW(int kernel_w) { this->primitive_->value.AsDeDepthwiseConv2D()->kernelW = kernel_w; }
void DeDepthwiseConv2D::SetKernelH(int kernel_h) { this->primitive_->value.AsDeDepthwiseConv2D()->kernelH = kernel_h; }
void DeDepthwiseConv2D::SetStrideW(int stride_w) { this->primitive_->value.AsDeDepthwiseConv2D()->strideW = stride_w; }
void DeDepthwiseConv2D::SetStrideH(int stride_h) { this->primitive_->value.AsDeDepthwiseConv2D()->strideH = stride_h; }
void DeDepthwiseConv2D::SetPadMode(int pad_mode) {
  this->primitive_->value.AsDeDepthwiseConv2D()->padMode = static_cast<schema::PadMode>(pad_mode);
}
void DeDepthwiseConv2D::SetPadUp(int pad_up) { this->primitive_->value.AsDeDepthwiseConv2D()->padUp = pad_up; }
void DeDepthwiseConv2D::SetPadDown(int pad_down) { this->primitive_->value.AsDeDepthwiseConv2D()->padDown = pad_down; }
void DeDepthwiseConv2D::SetPadLeft(int pad_left) { this->primitive_->value.AsDeDepthwiseConv2D()->padLeft = pad_left; }
void DeDepthwiseConv2D::SetPadRight(int pad_right) {
  this->primitive_->value.AsDeDepthwiseConv2D()->padRight = pad_right;
}
void DeDepthwiseConv2D::SetDilateW(int dilate_w) { this->primitive_->value.AsDeDepthwiseConv2D()->dilateW = dilate_w; }
void DeDepthwiseConv2D::SetDilateH(int dilate_h) { this->primitive_->value.AsDeDepthwiseConv2D()->dilateH = dilate_h; }
void DeDepthwiseConv2D::SetActivationType(int activation_type) {
  this->primitive_->value.AsDeDepthwiseConv2D()->activationType = static_cast<schema::ActivationType>(activation_type);
}

#else
int DeDepthwiseConv2D::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);

  auto attr = primitive->value_as_DeDepthwiseConv2D();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_DeDepthwiseConv2D return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateDeDepthwiseConv2D(
    *fbb, attr->format(), attr->channelIn(), attr->channelMultiplier(), attr->kernelW(), attr->kernelH(),
    attr->strideW(), attr->strideH(), attr->padMode(), attr->padUp(), attr->padDown(), attr->padLeft(),
    attr->padRight(), attr->dilateW(), attr->dilateH(), attr->hasBias(), attr->activationType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_DeDepthwiseConv2D, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int DeDepthwiseConv2D::GetFormat() const { return this->primitive_->value_as_DeDepthwiseConv2D()->format(); }
int DeDepthwiseConv2D::GetChannelIn() const { return this->primitive_->value_as_DeDepthwiseConv2D()->channelIn(); }
int DeDepthwiseConv2D::GetChannelMultiplier() const {
  return this->primitive_->value_as_DeDepthwiseConv2D()->channelMultiplier();
}
int DeDepthwiseConv2D::GetKernelW() const { return this->primitive_->value_as_DeDepthwiseConv2D()->kernelW(); }
int DeDepthwiseConv2D::GetKernelH() const { return this->primitive_->value_as_DeDepthwiseConv2D()->kernelH(); }
int DeDepthwiseConv2D::GetStrideW() const { return this->primitive_->value_as_DeDepthwiseConv2D()->strideW(); }
int DeDepthwiseConv2D::GetStrideH() const { return this->primitive_->value_as_DeDepthwiseConv2D()->strideH(); }
int DeDepthwiseConv2D::GetPadMode() const { return this->primitive_->value_as_DeDepthwiseConv2D()->padMode(); }
int DeDepthwiseConv2D::GetPadUp() const { return this->primitive_->value_as_DeDepthwiseConv2D()->padUp(); }
int DeDepthwiseConv2D::GetPadDown() const { return this->primitive_->value_as_DeDepthwiseConv2D()->padDown(); }
int DeDepthwiseConv2D::GetPadLeft() const { return this->primitive_->value_as_DeDepthwiseConv2D()->padLeft(); }
int DeDepthwiseConv2D::GetPadRight() const { return this->primitive_->value_as_DeDepthwiseConv2D()->padRight(); }
int DeDepthwiseConv2D::GetDilateW() const { return this->primitive_->value_as_DeDepthwiseConv2D()->dilateW(); }
int DeDepthwiseConv2D::GetDilateH() const { return this->primitive_->value_as_DeDepthwiseConv2D()->dilateH(); }
int DeDepthwiseConv2D::GetActivationType() const {
  return this->primitive_->value_as_DeDepthwiseConv2D()->activationType();
}

PrimitiveC *DeDepthwiseConv2DCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<DeDepthwiseConv2D>(primitive);
}
Registry DeDepthwiseConv2DRegistry(schema::PrimitiveType_DeDepthwiseConv2D, DeDepthwiseConv2DCreator);
#endif

int DeDepthwiseConv2D::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  if (inputs_.size() != kDoubleNum && inputs_.size() != kTripleNum) {
    MS_LOG(ERROR) << "inputs number is invalid";
    return 1;
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "output number is invalid";
    return 1;
  }
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto weight = inputs_.at(1);
  MS_ASSERT(weight != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_format(input->format());
  output->set_data_type(input->data_type());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto in_shape = input->shape();
  int input_h = in_shape.at(1);
  int input_w = in_shape.at(2);
  int input_channel = in_shape.at(3);
  int output_w = 0, output_h = 0;

  pad_l_ = GetPadLeft();
  pad_u_ = GetPadUp();
  pad_d_ = GetPadDown();
  pad_r_ = GetPadRight();
  output_h = GetStrideH() * (input_h - 1) + GetKernelH() - pad_u_ - pad_d_;
  output_w = GetStrideW() * (input_w - 1) + GetKernelW() - pad_l_ - pad_r_;
  if ((output_h + GetPadUp() + GetPadDown() - GetKernelH()) % GetStrideH() != 0) {
    output_h += (output_h + GetPadLeft() + GetPadRight() - GetKernelH()) % GetStrideH();
  }
  if ((output_w + GetPadLeft() + GetPadRight() - GetKernelW()) % GetStrideW() != 0) {
    output_w += (output_w + GetPadLeft() + GetPadRight() - GetKernelW()) % GetStrideW();
  }
  std::vector<int> out_shape{input->shape()};
  out_shape.at(1) = output_h;
  out_shape.at(2) = output_w;
  if (GetChannelMultiplier() * input_channel != weight->shape()[0]) {
    MS_LOG(ERROR) << "Conv dedepthwise only support group equals output channel.";
    return RET_ERROR;
  }
  out_shape.at(3) = weight->shape()[0] * weight->shape()[3];  // in_channel * out_channel

  output->set_shape(out_shape);
  return 0;
}
}  // namespace lite
}  // namespace mindspore
