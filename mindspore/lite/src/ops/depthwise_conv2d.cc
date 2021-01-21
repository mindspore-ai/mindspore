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

#include "src/ops/depthwise_conv2d.h"

#include <memory>
#include <string>
#ifdef PRIMITIVE_WRITEABLE
#include "src/param_value_lite.h"
#endif
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int DepthwiseConv2D::GetFormat() const { return this->primitive_->value.AsDepthwiseConv2D()->format; }
int DepthwiseConv2D::GetChannelIn() const { return this->primitive_->value.AsDepthwiseConv2D()->channelIn; }
int DepthwiseConv2D::GetChannelMultiplier() const {
  return this->primitive_->value.AsDepthwiseConv2D()->channelMultiplier;
}
int DepthwiseConv2D::GetKernelW() const { return this->primitive_->value.AsDepthwiseConv2D()->kernelW; }
int DepthwiseConv2D::GetKernelH() const { return this->primitive_->value.AsDepthwiseConv2D()->kernelH; }
int DepthwiseConv2D::GetStrideW() const { return this->primitive_->value.AsDepthwiseConv2D()->strideW; }
int DepthwiseConv2D::GetStrideH() const { return this->primitive_->value.AsDepthwiseConv2D()->strideH; }
int DepthwiseConv2D::GetPadMode() const { return this->primitive_->value.AsDepthwiseConv2D()->padMode; }
int DepthwiseConv2D::GetPadUp() const { return this->primitive_->value.AsDepthwiseConv2D()->padUp; }
int DepthwiseConv2D::GetPadDown() const { return this->primitive_->value.AsDepthwiseConv2D()->padDown; }
int DepthwiseConv2D::GetPadLeft() const { return this->primitive_->value.AsDepthwiseConv2D()->padLeft; }
int DepthwiseConv2D::GetPadRight() const { return this->primitive_->value.AsDepthwiseConv2D()->padRight; }
int DepthwiseConv2D::GetDilateW() const { return this->primitive_->value.AsDepthwiseConv2D()->dilateW; }
int DepthwiseConv2D::GetDilateH() const { return this->primitive_->value.AsDepthwiseConv2D()->dilateH; }
int DepthwiseConv2D::GetActivationType() const { return this->primitive_->value.AsDepthwiseConv2D()->activationType; }

void DepthwiseConv2D::SetFormat(int format) {
  this->primitive_->value.AsDepthwiseConv2D()->format = static_cast<schema::Format>(format);
}
void DepthwiseConv2D::SetChannelIn(int channel_in) {
  this->primitive_->value.AsDepthwiseConv2D()->channelIn = channel_in;
}
void DepthwiseConv2D::SetChannelMultiplier(int channel_multiplier) {
  this->primitive_->value.AsDepthwiseConv2D()->channelMultiplier = channel_multiplier;
}
void DepthwiseConv2D::SetKernelW(int kernel_w) { this->primitive_->value.AsDepthwiseConv2D()->kernelW = kernel_w; }
void DepthwiseConv2D::SetKernelH(int kernel_h) { this->primitive_->value.AsDepthwiseConv2D()->kernelH = kernel_h; }
void DepthwiseConv2D::SetStrideW(int stride_w) { this->primitive_->value.AsDepthwiseConv2D()->strideW = stride_w; }
void DepthwiseConv2D::SetStrideH(int stride_h) { this->primitive_->value.AsDepthwiseConv2D()->strideH = stride_h; }
void DepthwiseConv2D::SetPadMode(int pad_mode) {
  this->primitive_->value.AsDepthwiseConv2D()->padMode = static_cast<schema::PadMode>(pad_mode);
}
void DepthwiseConv2D::SetPadUp(int pad_up) { this->primitive_->value.AsDepthwiseConv2D()->padUp = pad_up; }
void DepthwiseConv2D::SetPadDown(int pad_down) { this->primitive_->value.AsDepthwiseConv2D()->padDown = pad_down; }
void DepthwiseConv2D::SetPadLeft(int pad_left) { this->primitive_->value.AsDepthwiseConv2D()->padLeft = pad_left; }
void DepthwiseConv2D::SetPadRight(int pad_right) { this->primitive_->value.AsDepthwiseConv2D()->padRight = pad_right; }
void DepthwiseConv2D::SetDilateW(int dilate_w) { this->primitive_->value.AsDepthwiseConv2D()->dilateW = dilate_w; }
void DepthwiseConv2D::SetDilateH(int dilate_h) { this->primitive_->value.AsDepthwiseConv2D()->dilateH = dilate_h; }
void DepthwiseConv2D::SetActivationType(int activation_type) {
  this->primitive_->value.AsDepthwiseConv2D()->activationType = static_cast<schema::ActivationType>(activation_type);
}

int DepthwiseConv2D::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  this->primitive_ = new (schema::PrimitiveT);
  auto attr = std::make_unique<schema::DepthwiseConv2DT>();

  auto format = GetValue<std::string>(prim.GetAttr("data_format"));
  if (format == "NCHW") {
    attr->format = schema::Format::Format_NCHW;
  } else if (format == "NHWC") {
    attr->format = schema::Format::Format_NHWC;
  } else {
    attr->format = schema::Format::Format_NUM_OF_FORMAT;
  }
  auto pad_list = CastToInt(prim.GetAttr("pads"));
  attr->padUp = pad_list.at(0);
  attr->padDown = pad_list.at(1);
  attr->padLeft = pad_list.at(2);
  attr->padRight = pad_list.at(3);

  auto dilation = CastToInt(prim.GetAttr("dilation"));
  attr->dilateH = dilation.at(0);
  attr->dilateW = dilation.at(1);

  if (utils::isa<ValueSequeue>(prim.GetAttr("kernel_size"))) {
    auto kernel_size = CastToInt(prim.GetAttr("kernel_size"));
    attr->kernelH = kernel_size.at(0);
    attr->kernelW = kernel_size.at(1);
  } else {
    auto kernel_size = CastToInt(prim.GetAttr("kernel_size")).front();
    attr->kernelH = kernel_size;
    attr->kernelW = kernel_size;
  }

  auto stride = CastToInt(prim.GetAttr("stride"));
  attr->strideH = stride.at(2);
  attr->strideW = stride.at(3);

  auto pad_mode = GetValue<std::string>(prim.GetAttr("pad_mode"));
  if (pad_mode == "valid") {
    attr->padMode = schema::PadMode_VALID;
  } else if (pad_mode == "same") {
    attr->padMode = schema::PadMode_SAME_UPPER;
  } else {
    attr->padMode = schema::PadMode_NOTSET;
  }
  if (prim.GetAttr("activation_name") != nullptr) {
    std::string activate_name = GetValue<std::string>(prim.GetAttr("activation_name"));
    attr->activationType = kActivationTypeMap[activate_name];
  } else {
    attr->activationType = schema::ActivationType_NO_ACTIVATION;
  }
  auto channel_multiplier = CastToInt(prim.GetAttr("channel_multiplier")).front();
  attr->channelMultiplier = channel_multiplier;

  MS_ASSERT(inputs.size() == kAnfPopulaterInputNumTwo);
  auto inputNode = inputs.at(kAnfPopulaterInputNumOne);
  MS_ASSERT(inputNode != nullptr);
  if (inputNode->isa<Parameter>()) {
    auto paramNode = inputNode->cast<ParameterPtr>();
    auto abstractBase = paramNode->abstract();
    MS_ASSERT(abstractBase != nullptr);
    if (utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
      auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
      MS_ASSERT(abstractTensor != nullptr);
      if (utils::isa<abstract::ShapePtr>(abstractTensor->BuildShape())) {
        auto dims = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
        attr->channelIn = dims.at(kAnfPopulaterInputNumOne);
      }
    }
  }

  this->primitive_->value.type = schema::PrimitiveType_DepthwiseConv2D;
  this->primitive_->value.value = attr.release();
  PopulaterQuantParam(prim, inputs);
  return RET_OK;
}

#else
int DepthwiseConv2D::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_DepthwiseConv2D();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_DepthwiseConv2D return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateDepthwiseConv2D(
    *fbb, attr->format(), attr->channelIn(), attr->channelMultiplier(), attr->kernelW(), attr->kernelH(),
    attr->strideW(), attr->strideH(), attr->padMode(), attr->padUp(), attr->padDown(), attr->padLeft(),
    attr->padRight(), attr->dilateW(), attr->dilateH(), attr->hasBias(), attr->activationType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_DepthwiseConv2D, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int DepthwiseConv2D::GetFormat() const { return this->primitive_->value_as_DepthwiseConv2D()->format(); }
int DepthwiseConv2D::GetChannelIn() const { return this->primitive_->value_as_DepthwiseConv2D()->channelIn(); }
int DepthwiseConv2D::GetChannelMultiplier() const {
  return this->primitive_->value_as_DepthwiseConv2D()->channelMultiplier();
}
int DepthwiseConv2D::GetKernelW() const { return this->primitive_->value_as_DepthwiseConv2D()->kernelW(); }
int DepthwiseConv2D::GetKernelH() const { return this->primitive_->value_as_DepthwiseConv2D()->kernelH(); }
int DepthwiseConv2D::GetStrideW() const { return this->primitive_->value_as_DepthwiseConv2D()->strideW(); }
int DepthwiseConv2D::GetStrideH() const { return this->primitive_->value_as_DepthwiseConv2D()->strideH(); }
int DepthwiseConv2D::GetPadMode() const { return this->primitive_->value_as_DepthwiseConv2D()->padMode(); }
int DepthwiseConv2D::GetPadUp() const { return this->primitive_->value_as_DepthwiseConv2D()->padUp(); }
int DepthwiseConv2D::GetPadDown() const { return this->primitive_->value_as_DepthwiseConv2D()->padDown(); }
int DepthwiseConv2D::GetPadLeft() const { return this->primitive_->value_as_DepthwiseConv2D()->padLeft(); }
int DepthwiseConv2D::GetPadRight() const { return this->primitive_->value_as_DepthwiseConv2D()->padRight(); }
int DepthwiseConv2D::GetDilateW() const { return this->primitive_->value_as_DepthwiseConv2D()->dilateW(); }
int DepthwiseConv2D::GetDilateH() const { return this->primitive_->value_as_DepthwiseConv2D()->dilateH(); }
int DepthwiseConv2D::GetActivationType() const {
  return this->primitive_->value_as_DepthwiseConv2D()->activationType();
}

PrimitiveC *DepthWiseConv2DCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<DepthwiseConv2D>(primitive);
}
Registry DepthWiseConv2DRegistry(schema::PrimitiveType_DepthwiseConv2D, DepthWiseConv2DCreator);

#endif

int DepthwiseConv2D::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
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
  pad_l_ = GetPadLeft();
  pad_u_ = GetPadUp();
  pad_d_ = GetPadDown();
  pad_r_ = GetPadRight();

  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto in_shape = input->shape();
  int input_h = in_shape.at(1);
  int input_w = in_shape.at(2);
  int input_channel = in_shape.at(3);
  int output_w = 0, output_h = 0;
  input_channel_ = input_channel;

  if (GetPadMode() == schema::PadMode_SAME_UPPER) {
    output_h = std::ceil(static_cast<float>(input_h) / static_cast<float>(GetStrideH()));
    output_w = std::ceil(static_cast<float>(input_w) / static_cast<float>(GetStrideW()));
    auto pad_h_all = ((output_h - 1) * GetStrideH() + (GetKernelH() - 1) * GetDilateH() + 1 - input_h);
    auto pad_w_all = ((output_w - 1) * GetStrideW() + (GetKernelW() - 1) * GetDilateW() + 1 - input_w);
    if (pad_h_all > 0) {
      pad_u_ = pad_h_all / 2;
      pad_d_ = pad_h_all - pad_u_;
    }
    if (pad_w_all > 0) {
      pad_l_ = pad_w_all / 2;
      pad_r_ = pad_w_all - pad_l_;
    }
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
  if (GetChannelMultiplier() * input_channel != weight->shape().at(0)) {
    MS_LOG(ERROR) << "Conv depthwise only support group equals output channel.";
    return 1;
  }
  out_shape.at(3) = weight->shape().at(0) * weight->shape().at(3);  // in_channel * out_channel

  output->set_shape(out_shape);
  return 0;
}
}  // namespace lite
}  // namespace mindspore
