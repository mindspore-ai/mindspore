/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/ops/adder.h"
#include <memory>
#include <string>

#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#ifdef PRIMITIVE_WRITEABLE
#include "src/param_value_lite.h"
#endif

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Adder::GetFormat() const { return this->primitive_->value.AsAdder()->format; }
int Adder::GetGroup() const { return this->primitive_->value.AsAdder()->group; }
int Adder::GetChannelIn() const { return this->primitive_->value.AsAdder()->channelIn; }
int Adder::GetChannelOut() const { return this->primitive_->value.AsAdder()->channelOut; }
int Adder::GetKernelW() const { return this->primitive_->value.AsAdder()->kernelW; }
int Adder::GetKernelH() const { return this->primitive_->value.AsAdder()->kernelH; }
int Adder::GetStrideW() const { return this->primitive_->value.AsAdder()->strideW; }
int Adder::GetStrideH() const { return this->primitive_->value.AsAdder()->strideH; }
int Adder::GetPadMode() const { return this->primitive_->value.AsAdder()->padMode; }
int Adder::GetPadUp() const { return this->primitive_->value.AsAdder()->padUp; }
int Adder::GetPadDown() const { return this->primitive_->value.AsAdder()->padDown; }
int Adder::GetPadLeft() const { return this->primitive_->value.AsAdder()->padLeft; }
int Adder::GetPadRight() const { return this->primitive_->value.AsAdder()->padRight; }
int Adder::GetDilateW() const { return this->primitive_->value.AsAdder()->dilateW; }
int Adder::GetDilateH() const { return this->primitive_->value.AsAdder()->dilateH; }
int Adder::GetActivationType() const { return this->primitive_->value.AsAdder()->activationType; }

void Adder::SetFormat(int format) { this->primitive_->value.AsAdder()->format = (schema::Format)format; }
void Adder::SetGroup(int group) { this->primitive_->value.AsAdder()->group = group; }
void Adder::SetChannelIn(int channel_in) { this->primitive_->value.AsAdder()->channelIn = channel_in; }
void Adder::SetChannelOut(int channel_out) { this->primitive_->value.AsAdder()->channelOut = channel_out; }
void Adder::SetKernelW(int kernel_w) { this->primitive_->value.AsAdder()->kernelW = kernel_w; }
void Adder::SetKernelH(int kernel_h) { this->primitive_->value.AsAdder()->kernelH = kernel_h; }
void Adder::SetStrideW(int stride_w) { this->primitive_->value.AsAdder()->strideW = stride_w; }
void Adder::SetStrideH(int stride_h) { this->primitive_->value.AsAdder()->strideH = stride_h; }
void Adder::SetPadMode(int pad_mode) { this->primitive_->value.AsAdder()->padMode = (schema::PadMode)pad_mode; }
void Adder::SetPadUp(int pad_up) { this->primitive_->value.AsAdder()->padUp = pad_up; }
void Adder::SetPadDown(int pad_down) { this->primitive_->value.AsAdder()->padDown = pad_down; }
void Adder::SetPadLeft(int pad_left) { this->primitive_->value.AsAdder()->padLeft = pad_left; }
void Adder::SetPadRight(int pad_right) { this->primitive_->value.AsAdder()->padRight = pad_right; }
void Adder::SetDilateW(int dilate_w) { this->primitive_->value.AsAdder()->dilateW = dilate_w; }
void Adder::SetDilateH(int dilate_h) { this->primitive_->value.AsAdder()->dilateH = dilate_h; }
void Adder::SetActivationType(int activation_type) {
  this->primitive_->value.AsAdder()->activationType = (schema::ActivationType)activation_type;
}

void Adder::PopulaterAdderSingleGroup(const Primitive &prim, schema::PrimitiveT *primitive, const int &group) {
  auto attr = std::make_unique<schema::AdderT>();
  attr->group = group;
  auto format = GetValue<std::string>(prim.GetAttr("data_format"));
  if (format == "NCHW") {
    attr->format = schema::Format::Format_NCHW;
  } else if (format == "NHWC") {
    attr->format = schema::Format::Format_NHWC;
  } else {
    attr->format = schema::Format::Format_NUM_OF_FORMAT;
  }
  auto pad_list = CastToInt(prim.GetAttr("pad_list"));
  attr->padUp = pad_list[0];
  attr->padDown = pad_list[1];
  attr->padLeft = pad_list[2];
  attr->padRight = pad_list[3];

  auto dilation = CastToInt(prim.GetAttr("dilation"));
  attr->dilateH = dilation[2];
  attr->dilateW = dilation[3];

  auto kernel_size = CastToInt(prim.GetAttr("kernel_size"));
  attr->kernelH = kernel_size[0];
  attr->kernelW = kernel_size[1];

  auto stride = CastToInt(prim.GetAttr("stride"));
  attr->strideH = stride[2];
  attr->strideW = stride[3];

  attr->channelOut = CastToInt(prim.GetAttr("out_channel")).front();

  auto pad_mode = GetValue<std::string>(prim.GetAttr("pad_mode"));
  if (pad_mode == "valid") {
    attr->padMode = schema::PadMode_VALID;
  } else if (pad_mode == "same") {
    attr->padMode = schema::PadMode_SAME_UPPER;
  } else {
    attr->padMode = schema::PadMode_NOTSET;
  }

  if (prim.GetAttr("activation_name") != nullptr) {
    auto activate_name = GetValue<std::string>(prim.GetAttr("activation_name"));
    attr->activationType = kActivationTypeMap[activate_name];
  } else {
    attr->activationType = schema::ActivationType_NO_ACTIVATION;
  }

  primitive->value.type = schema::PrimitiveType_Adder;
  primitive->value.value = attr.release();
}

int Adder::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Adder;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Adder) {
    MS_LOG(ERROR) << "primitive_ type is error:" << this->primitive_->value.type;
    return RET_ERROR;
  }
  auto groupAttr = prim.GetAttr("group");
  if (groupAttr == nullptr) {
    MS_LOG(ERROR) << "conv2d op has no group attr,please check pb model";
    return RET_NULL_PTR;
  }
  int group = CastToInt(groupAttr).front();
  PopulaterAdderSingleGroup(prim, this->primitive_, group);
  PopulaterQuantParam(prim, inputs);
  return RET_OK;
}

#else
int Adder::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Adder();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Adder return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateAdder(*fbb, attr->format(), attr->group(), attr->channelIn(), attr->channelOut(),
                                        attr->kernelW(), attr->kernelH(), attr->strideW(), attr->strideH(),
                                        attr->padMode(), attr->padUp(), attr->padDown(), attr->padLeft(),
                                        attr->padRight(), attr->dilateW(), attr->dilateH(), attr->activationType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Adder, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

int Adder::GetFormat() const { return this->primitive_->value_as_Adder()->format(); }
int Adder::GetGroup() const { return this->primitive_->value_as_Adder()->group(); }
int Adder::GetChannelIn() const { return this->primitive_->value_as_Adder()->channelIn(); }
int Adder::GetChannelOut() const { return this->primitive_->value_as_Adder()->channelOut(); }
int Adder::GetKernelW() const { return this->primitive_->value_as_Adder()->kernelW(); }
int Adder::GetKernelH() const { return this->primitive_->value_as_Adder()->kernelH(); }
int Adder::GetStrideW() const { return this->primitive_->value_as_Adder()->strideW(); }
int Adder::GetStrideH() const { return this->primitive_->value_as_Adder()->strideH(); }
int Adder::GetPadMode() const { return this->primitive_->value_as_Adder()->padMode(); }
int Adder::GetPadUp() const { return this->primitive_->value_as_Adder()->padUp(); }
int Adder::GetPadDown() const { return this->primitive_->value_as_Adder()->padDown(); }
int Adder::GetPadLeft() const { return this->primitive_->value_as_Adder()->padLeft(); }
int Adder::GetPadRight() const { return this->primitive_->value_as_Adder()->padRight(); }
int Adder::GetDilateW() const { return this->primitive_->value_as_Adder()->dilateW(); }
int Adder::GetDilateH() const { return this->primitive_->value_as_Adder()->dilateH(); }
int Adder::GetActivationType() const { return this->primitive_->value_as_Adder()->activationType(); }

PrimitiveC *AdderCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Adder>(primitive); }
Registry AdderRegistry(schema::PrimitiveType_Adder, AdderCreator);
#endif
}  // namespace lite
}  // namespace mindspore
