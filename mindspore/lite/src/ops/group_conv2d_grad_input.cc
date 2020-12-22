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

#include "src/ops/group_conv2d_grad_input.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int GroupConv2DGradInput::GetFormat() const { return this->primitive_->value.AsGroupConv2DGradInput()->format; }
int GroupConv2DGradInput::GetGroup() const { return this->primitive_->value.AsGroupConv2DGradInput()->group; }
int GroupConv2DGradInput::GetChannelIn() const { return this->primitive_->value.AsGroupConv2DGradInput()->channelIn; }
int GroupConv2DGradInput::GetChannelOut() const { return this->primitive_->value.AsGroupConv2DGradInput()->channelOut; }
int GroupConv2DGradInput::GetKernelW() const { return this->primitive_->value.AsGroupConv2DGradInput()->kernelW; }
int GroupConv2DGradInput::GetKernelH() const { return this->primitive_->value.AsGroupConv2DGradInput()->kernelH; }
int GroupConv2DGradInput::GetStrideW() const { return this->primitive_->value.AsGroupConv2DGradInput()->strideW; }
int GroupConv2DGradInput::GetStrideH() const { return this->primitive_->value.AsGroupConv2DGradInput()->strideH; }
int GroupConv2DGradInput::GetPadMode() const { return this->primitive_->value.AsGroupConv2DGradInput()->padMode; }
int GroupConv2DGradInput::GetPadUp() const { return this->primitive_->value.AsGroupConv2DGradInput()->padUp; }
int GroupConv2DGradInput::GetPadDown() const { return this->primitive_->value.AsGroupConv2DGradInput()->padDown; }
int GroupConv2DGradInput::GetPadLeft() const { return this->primitive_->value.AsGroupConv2DGradInput()->padLeft; }
int GroupConv2DGradInput::GetPadRight() const { return this->primitive_->value.AsGroupConv2DGradInput()->padRight; }
int GroupConv2DGradInput::GetDilateW() const { return this->primitive_->value.AsGroupConv2DGradInput()->dilateW; }
int GroupConv2DGradInput::GetDilateH() const { return this->primitive_->value.AsGroupConv2DGradInput()->dilateH; }
std::vector<int> GroupConv2DGradInput::GetInputShape() const {
  return this->primitive_->value.AsGroupConv2DGradInput()->input_shape;
}
int GroupConv2DGradInput::GetActivationType() const {
  return this->primitive_->value.AsGroupConv2DGradInput()->activationType;
}

void GroupConv2DGradInput::SetFormat(int format) {
  this->primitive_->value.AsGroupConv2DGradInput()->format = (schema::Format)format;
}
void GroupConv2DGradInput::SetGroup(int group) { this->primitive_->value.AsGroupConv2DGradInput()->group = group; }
void GroupConv2DGradInput::SetChannelIn(int channel_in) {
  this->primitive_->value.AsGroupConv2DGradInput()->channelIn = channel_in;
}
void GroupConv2DGradInput::SetChannelOut(int channel_out) {
  this->primitive_->value.AsGroupConv2DGradInput()->channelOut = channel_out;
}
void GroupConv2DGradInput::SetKernelW(int kernel_w) {
  this->primitive_->value.AsGroupConv2DGradInput()->kernelW = kernel_w;
}
void GroupConv2DGradInput::SetKernelH(int kernel_h) {
  this->primitive_->value.AsGroupConv2DGradInput()->kernelH = kernel_h;
}
void GroupConv2DGradInput::SetStrideW(int stride_w) {
  this->primitive_->value.AsGroupConv2DGradInput()->strideW = stride_w;
}
void GroupConv2DGradInput::SetStrideH(int stride_h) {
  this->primitive_->value.AsGroupConv2DGradInput()->strideH = stride_h;
}
void GroupConv2DGradInput::SetPadMode(int pad_mode) {
  this->primitive_->value.AsGroupConv2DGradInput()->padMode = (schema::PadMode)pad_mode;
}
void GroupConv2DGradInput::SetPadUp(int pad_up) { this->primitive_->value.AsGroupConv2DGradInput()->padUp = pad_up; }
void GroupConv2DGradInput::SetPadDown(int pad_down) {
  this->primitive_->value.AsGroupConv2DGradInput()->padDown = pad_down;
}
void GroupConv2DGradInput::SetPadLeft(int pad_left) {
  this->primitive_->value.AsGroupConv2DGradInput()->padLeft = pad_left;
}
void GroupConv2DGradInput::SetPadRight(int pad_right) {
  this->primitive_->value.AsGroupConv2DGradInput()->padRight = pad_right;
}
void GroupConv2DGradInput::SetDilateW(int dilate_w) {
  this->primitive_->value.AsGroupConv2DGradInput()->dilateW = dilate_w;
}
void GroupConv2DGradInput::SetDilateH(int dilate_h) {
  this->primitive_->value.AsGroupConv2DGradInput()->dilateH = dilate_h;
}
void GroupConv2DGradInput::SetActivationType(int activation_type) {
  this->primitive_->value.AsGroupConv2DGradInput()->activationType = (schema::ActivationType)activation_type;
}
#else
int GroupConv2DGradInput::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_GroupConv2DGradInput();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_GroupConv2DGradInput return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> input_shape;
  if (attr->input_shape() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->input_shape()->size()); i++) {
      input_shape.push_back(attr->input_shape()->data()[i]);
    }
  }
  auto val_offset = schema::CreateGroupConv2DGradInputDirect(
    *fbb, attr->format(), attr->group(), attr->channelIn(), attr->channelOut(), attr->kernelW(), attr->kernelH(),
    attr->strideW(), attr->strideH(), attr->padMode(), attr->padUp(), attr->padDown(), attr->padLeft(),
    attr->padRight(), attr->dilateW(), attr->dilateH(), attr->hasBias(), &input_shape, attr->activationType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_GroupConv2DGradInput, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int GroupConv2DGradInput::GetFormat() const { return this->primitive_->value_as_GroupConv2DGradInput()->format(); }
int GroupConv2DGradInput::GetGroup() const { return this->primitive_->value_as_GroupConv2DGradInput()->group(); }
int GroupConv2DGradInput::GetChannelIn() const {
  return this->primitive_->value_as_GroupConv2DGradInput()->channelIn();
}
int GroupConv2DGradInput::GetChannelOut() const {
  return this->primitive_->value_as_GroupConv2DGradInput()->channelOut();
}
int GroupConv2DGradInput::GetKernelW() const { return this->primitive_->value_as_GroupConv2DGradInput()->kernelW(); }
int GroupConv2DGradInput::GetKernelH() const { return this->primitive_->value_as_GroupConv2DGradInput()->kernelH(); }
int GroupConv2DGradInput::GetStrideW() const { return this->primitive_->value_as_GroupConv2DGradInput()->strideW(); }
int GroupConv2DGradInput::GetStrideH() const { return this->primitive_->value_as_GroupConv2DGradInput()->strideH(); }
int GroupConv2DGradInput::GetPadMode() const { return this->primitive_->value_as_GroupConv2DGradInput()->padMode(); }
int GroupConv2DGradInput::GetPadUp() const { return this->primitive_->value_as_GroupConv2DGradInput()->padUp(); }
int GroupConv2DGradInput::GetPadDown() const { return this->primitive_->value_as_GroupConv2DGradInput()->padDown(); }
int GroupConv2DGradInput::GetPadLeft() const { return this->primitive_->value_as_GroupConv2DGradInput()->padLeft(); }
int GroupConv2DGradInput::GetPadRight() const { return this->primitive_->value_as_GroupConv2DGradInput()->padRight(); }
int GroupConv2DGradInput::GetDilateW() const { return this->primitive_->value_as_GroupConv2DGradInput()->dilateW(); }
int GroupConv2DGradInput::GetDilateH() const { return this->primitive_->value_as_GroupConv2DGradInput()->dilateH(); }
std::vector<int> GroupConv2DGradInput::GetInputShape() const {
  auto fb_vector = this->primitive_->value_as_GroupConv2DGradInput()->input_shape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int GroupConv2DGradInput::GetActivationType() const {
  return this->primitive_->value_as_GroupConv2DGradInput()->activationType();
}
PrimitiveC *GroupConv2DGradInputCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<GroupConv2DGradInput>(primitive);
}
Registry GroupConv2DGradInputRegistry(schema::PrimitiveType_GroupConv2DGradInput, GroupConv2DGradInputCreator);

#endif

int GroupConv2DGradInput::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  if (inputs.size() < 2) {
    MS_LOG(ERROR) << "Conv2d Grad input should be at least two input";
    return RET_ERROR;
  }
  if (1 != outputs.size()) {
    MS_LOG(ERROR) << "Conv2d Grad output should have one output";
    return RET_ERROR;
  }

  auto *in0 = inputs.at(0);

  MS_ASSERT(in0 != nullptr);

  auto *out = outputs.at(0);
  MS_ASSERT(out != nullptr);
  out->set_shape(GetInputShape());

  out->set_data_type(in0->data_type());
  out->set_format(in0->format());

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
