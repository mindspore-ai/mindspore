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

#include "src/ops/conv2d_grad_filter.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Conv2DGradFilter::GetFormat() const { return this->primitive_->value.AsConv2DGradFilter()->format; }
int Conv2DGradFilter::GetGroup() const { return this->primitive_->value.AsConv2DGradFilter()->group; }
int Conv2DGradFilter::GetChannelIn() const { return this->primitive_->value.AsConv2DGradFilter()->channelIn; }
int Conv2DGradFilter::GetChannelOut() const { return this->primitive_->value.AsConv2DGradFilter()->channelOut; }
int Conv2DGradFilter::GetKernelW() const { return this->primitive_->value.AsConv2DGradFilter()->kernelW; }
int Conv2DGradFilter::GetKernelH() const { return this->primitive_->value.AsConv2DGradFilter()->kernelH; }
int Conv2DGradFilter::GetStrideW() const { return this->primitive_->value.AsConv2DGradFilter()->strideW; }
int Conv2DGradFilter::GetStrideH() const { return this->primitive_->value.AsConv2DGradFilter()->strideH; }
int Conv2DGradFilter::GetPadMode() const { return this->primitive_->value.AsConv2DGradFilter()->padMode; }
int Conv2DGradFilter::GetPadUp() const { return this->primitive_->value.AsConv2DGradFilter()->padUp; }
int Conv2DGradFilter::GetPadDown() const { return this->primitive_->value.AsConv2DGradFilter()->padDown; }
int Conv2DGradFilter::GetPadLeft() const { return this->primitive_->value.AsConv2DGradFilter()->padLeft; }
int Conv2DGradFilter::GetPadRight() const { return this->primitive_->value.AsConv2DGradFilter()->padRight; }
int Conv2DGradFilter::GetDilateW() const { return this->primitive_->value.AsConv2DGradFilter()->dilateW; }
int Conv2DGradFilter::GetDilateH() const { return this->primitive_->value.AsConv2DGradFilter()->dilateH; }
bool Conv2DGradFilter::GetHasBias() const { return this->primitive_->value.AsConv2DGradFilter()->hasBias; }
int Conv2DGradFilter::GetActivationType() const { return this->primitive_->value.AsConv2DGradFilter()->activationType; }

void Conv2DGradFilter::SetFormat(int format) {
  this->primitive_->value.AsConv2DGradFilter()->format = (schema::Format)format;
}
void Conv2DGradFilter::SetGroup(int group) { this->primitive_->value.AsConv2DGradFilter()->group = group; }
void Conv2DGradFilter::SetChannelIn(int channel_in) {
  this->primitive_->value.AsConv2DGradFilter()->channelIn = channel_in;
}
void Conv2DGradFilter::SetChannelOut(int channel_out) {
  this->primitive_->value.AsConv2DGradFilter()->channelOut = channel_out;
}
void Conv2DGradFilter::SetKernelW(int kernel_w) { this->primitive_->value.AsConv2DGradFilter()->kernelW = kernel_w; }
void Conv2DGradFilter::SetKernelH(int kernel_h) { this->primitive_->value.AsConv2DGradFilter()->kernelH = kernel_h; }
void Conv2DGradFilter::SetStrideW(int stride_w) { this->primitive_->value.AsConv2DGradFilter()->strideW = stride_w; }
void Conv2DGradFilter::SetStrideH(int stride_h) { this->primitive_->value.AsConv2DGradFilter()->strideH = stride_h; }
void Conv2DGradFilter::SetPadMode(int pad_mode) {
  this->primitive_->value.AsConv2DGradFilter()->padMode = (schema::PadMode)pad_mode;
}
void Conv2DGradFilter::SetPadUp(int pad_up) { this->primitive_->value.AsConv2DGradFilter()->padUp = pad_up; }
void Conv2DGradFilter::SetPadDown(int pad_down) { this->primitive_->value.AsConv2DGradFilter()->padDown = pad_down; }
void Conv2DGradFilter::SetPadLeft(int pad_left) { this->primitive_->value.AsConv2DGradFilter()->padLeft = pad_left; }
void Conv2DGradFilter::SetPadRight(int pad_right) {
  this->primitive_->value.AsConv2DGradFilter()->padRight = pad_right;
}
void Conv2DGradFilter::SetDilateW(int dilate_w) { this->primitive_->value.AsConv2DGradFilter()->dilateW = dilate_w; }
void Conv2DGradFilter::SetDilateH(int dilate_h) { this->primitive_->value.AsConv2DGradFilter()->dilateH = dilate_h; }
void Conv2DGradFilter::SetHasBias(bool has_bias) { this->primitive_->value.AsConv2DGradFilter()->hasBias = has_bias; }
void Conv2DGradFilter::SetActivationType(int activation_type) {
  this->primitive_->value.AsConv2DGradFilter()->activationType = (schema::ActivationType)activation_type;
}
void Conv2DGradFilter::PopulaterConv2DMultiGroup(const Primitive &prim, schema::PrimitiveT *primitive, const int &group,
                                       const std::vector<AnfNodePtr> &inputs) {
  auto attr = std::make_unique<schema::DepthwiseConv2DT>();
  auto format = GetValue<std::string>(prim.GetAttr("data_format"));
  if (format == "NCHW") {
    attr->format = schema::Format_NCHW;
  } else if (format == "NHWC") {
    attr->format = schema::Format_NHWC;
  } else {
    attr->format = schema::Format_NUM_OF_FORMAT;
  }
  auto pad_list = GetValue<std::vector<int>>(prim.GetAttr("pad_list"));
  attr->padUp = pad_list[0];
  attr->padDown = pad_list[1];
  attr->padLeft = pad_list[2];
  attr->padRight = pad_list[3];

  auto dilation = GetValue<std::vector<int>>(prim.GetAttr("dilation"));
  attr->dilateH = dilation[0];
  attr->dilateW = dilation[1];

  auto kernel_size = GetValue<std::vector<int>>(prim.GetAttr("kernel_size"));
  attr->kernelH = kernel_size[0];
  attr->kernelW = kernel_size[1];

  auto stride = GetValue<std::vector<int>>(prim.GetAttr("stride"));
  attr->strideH = stride[2];
  attr->strideW = stride[3];

  auto pad_mode = GetValue<std::string>(prim.GetAttr("pad_mode"));
  if (pad_mode == "valid") {
    attr->padMode = schema::PadMode_VALID;
  } else if (pad_mode == "same") {
    attr->padMode = schema::PadMode_SAME;
  } else {
    attr->padMode = schema::PadMode_NOTSET;
  }

  if (prim.GetAttr("activation_name") != nullptr) {
    std::string activate_name = GetValue<std::string>(prim.GetAttr("activation_name"));
    attr->activationType = kActivationTypeMap[activate_name];
  } else {
    attr->activationType = schema::ActivationType_NO_ACTIVATION;
  }

  int channel_mutiplier = 1;
  if (prim.GetAttr("channel_mutiplier") != nullptr) {
    channel_mutiplier = GetValue<int>(prim.GetAttr("channel_multiplier"));
  }
  attr->channelMultiplier = channel_mutiplier;
  primitive->value.value = attr.release();
}

void Conv2DGradFilter::PopulaterConv2DSingleGroup(const Primitive &prim,
  schema::PrimitiveT *primitive, const int &group) {
  auto attr = std::make_unique<schema::Conv2DT>();
  attr->group = group;
  auto format = GetValue<std::string>(prim.GetAttr("data_format"));
  if (format == "NCHW") {
    attr->format = schema::Format_NCHW;
  } else if (format == "NHWC") {
    attr->format = schema::Format_NHWC;
  } else {
    attr->format = schema::Format_NUM_OF_FORMAT;
  }
  auto pad_list = GetValue<std::vector<int>>(prim.GetAttr("pad_list"));
  attr->padUp = pad_list[0];
  attr->padDown = pad_list[1];
  attr->padLeft = pad_list[2];
  attr->padRight = pad_list[3];

  auto dilation = GetValue<std::vector<int>>(prim.GetAttr("dilation"));
  attr->dilateH = dilation[0];
  attr->dilateW = dilation[1];

  auto kernel_size = GetValue<std::vector<int>>(prim.GetAttr("kernel_size"));
  attr->kernelH = kernel_size[0];
  attr->kernelW = kernel_size[1];

  auto stride = GetValue<std::vector<int>>(prim.GetAttr("stride"));
  attr->strideH = stride[2];
  attr->strideW = stride[3];

  attr->channelOut = GetValue<int>(prim.GetAttr("out_channel"));

  auto pad_mode = GetValue<std::string>(prim.GetAttr("pad_mode"));
  if (pad_mode == "valid") {
    attr->padMode = schema::PadMode_VALID;
  } else if (pad_mode == "same") {
    attr->padMode = schema::PadMode_SAME;
  } else {
    attr->padMode = schema::PadMode_NOTSET;
  }

  if (prim.GetAttr("activation_name") != nullptr) {
    std::string activate_name = GetValue<std::string>(prim.GetAttr("activation_name"));
    attr->activationType = kActivationTypeMap[activate_name];
  } else {
    attr->activationType = schema::ActivationType_NO_ACTIVATION;
  }
  primitive->value.value = attr.release();
}
int Conv2DGradFilter::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Conv2DGradFilter;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Conv2DGradFilter) {
    MS_LOG(ERROR) << "primitive_ type is error:" << this->primitive_->value.type;
    return RET_ERROR;
  }
  int group = GetValue<int>(prim.GetAttr("group"));
  if (group > 1) {
    PopulaterConv2DMultiGroup(prim, this->primitive_, group, inputs);
  } else {
    PopulaterConv2DSingleGroup(prim, this->primitive_, group);
  }
  return RET_OK;
}
#else
int Conv2DGradFilter::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Conv2DGradFilter();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Conv2DGradFilter return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateConv2DGradFilter(
    *fbb, attr->format(), attr->group(), attr->channelIn(), attr->channelOut(), attr->kernelW(), attr->kernelH(),
    attr->strideW(), attr->strideH(), attr->padMode(), attr->padUp(), attr->padDown(), attr->padLeft(),
    attr->padRight(), attr->dilateW(), attr->dilateH(), attr->hasBias(), attr->activationType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Conv2DGradFilter, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int Conv2DGradFilter::GetFormat() const { return this->primitive_->value_as_Conv2DGradFilter()->format(); }
int Conv2DGradFilter::GetGroup() const { return this->primitive_->value_as_Conv2DGradFilter()->group(); }
int Conv2DGradFilter::GetChannelIn() const { return this->primitive_->value_as_Conv2DGradFilter()->channelIn(); }
int Conv2DGradFilter::GetChannelOut() const { return this->primitive_->value_as_Conv2DGradFilter()->channelOut(); }
int Conv2DGradFilter::GetKernelW() const { return this->primitive_->value_as_Conv2DGradFilter()->kernelW(); }
int Conv2DGradFilter::GetKernelH() const { return this->primitive_->value_as_Conv2DGradFilter()->kernelH(); }
int Conv2DGradFilter::GetStrideW() const { return this->primitive_->value_as_Conv2DGradFilter()->strideW(); }
int Conv2DGradFilter::GetStrideH() const { return this->primitive_->value_as_Conv2DGradFilter()->strideH(); }
int Conv2DGradFilter::GetPadMode() const { return this->primitive_->value_as_Conv2DGradFilter()->padMode(); }
int Conv2DGradFilter::GetPadUp() const { return this->primitive_->value_as_Conv2DGradFilter()->padUp(); }
int Conv2DGradFilter::GetPadDown() const { return this->primitive_->value_as_Conv2DGradFilter()->padDown(); }
int Conv2DGradFilter::GetPadLeft() const { return this->primitive_->value_as_Conv2DGradFilter()->padLeft(); }
int Conv2DGradFilter::GetPadRight() const { return this->primitive_->value_as_Conv2DGradFilter()->padRight(); }
int Conv2DGradFilter::GetDilateW() const { return this->primitive_->value_as_Conv2DGradFilter()->dilateW(); }
int Conv2DGradFilter::GetDilateH() const { return this->primitive_->value_as_Conv2DGradFilter()->dilateH(); }
bool Conv2DGradFilter::GetHasBias() const { return this->primitive_->value_as_Conv2DGradFilter()->hasBias(); }
int Conv2DGradFilter::GetActivationType() const {
  return this->primitive_->value_as_Conv2DGradFilter()->activationType();
}

#endif

int Conv2DGradFilter::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  if (3 != inputs.size()) {
    MS_LOG(ERROR) << "Conv2d Grad Filter should have 3 inputs";
    return RET_ERROR;
  }
  if (1 != outputs.size()) {
    MS_LOG(ERROR) << "Conv2d Grad Filter should have one output";
    return RET_ERROR;
  }

  auto *in0 = inputs.at(0);
  auto *in = inputs.at(2);
  MS_ASSERT(out != nullptr);

  std::vector<int> output_shape;
  int *out_shape = reinterpret_cast<int *>(in->Data());
  int new_size = in->ElementsNum();
  if (in0->GetFormat() == in->GetFormat()) {
    for (int i = 0; i < new_size; i++) output_shape.push_back(out_shape[i]);
  } else {
    if ((in0->GetFormat() == schema::Format_NHWC) && (in->GetFormat() == schema::Format_NCHW)) {
      output_shape.push_back(out_shape[0]);
      output_shape.push_back(out_shape[2]);
      output_shape.push_back(out_shape[3]);
      output_shape.push_back(out_shape[1]);
    } else {
      MS_LOG(ERROR) << "Shape covnert is not supported";
      return RET_ERROR;
    }
  }

  auto *out = outputs.at(0);
  MS_ASSERT(out != nullptr);

  out->set_shape(output_shape);
  out->set_data_type(in0->data_type());
  out->SetFormat(in0->GetFormat());

  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore
