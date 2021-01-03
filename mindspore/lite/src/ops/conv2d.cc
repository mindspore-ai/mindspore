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

#include "src/ops/conv2d.h"

#include <map>
#include <memory>
#include <string>

#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#ifdef PRIMITIVE_WRITEABLE
#include <float.h>
#include "src/param_value_lite.h"
#endif

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
int Conv2D::PadUp() const { return this->pad_u_; }
int Conv2D::PadDown() const { return this->pad_d_; }
int Conv2D::PadLeft() const { return this->pad_l_; }
int Conv2D::PadRight() const { return this->pad_r_; }
#ifdef PRIMITIVE_WRITEABLE
int Conv2D::GetFormat() const { return this->primitive_->value.AsConv2D()->format; }
int Conv2D::GetGroup() const { return this->primitive_->value.AsConv2D()->group; }
int Conv2D::GetChannelIn() const { return this->primitive_->value.AsConv2D()->channelIn; }
int Conv2D::GetChannelOut() const { return this->primitive_->value.AsConv2D()->channelOut; }
int Conv2D::GetKernelW() const { return this->primitive_->value.AsConv2D()->kernelW; }
int Conv2D::GetKernelH() const { return this->primitive_->value.AsConv2D()->kernelH; }
int Conv2D::GetStrideW() const { return this->primitive_->value.AsConv2D()->strideW; }
int Conv2D::GetStrideH() const { return this->primitive_->value.AsConv2D()->strideH; }
int Conv2D::GetPadMode() const { return this->primitive_->value.AsConv2D()->padMode; }
int Conv2D::GetPadUp() const { return this->primitive_->value.AsConv2D()->padUp; }
int Conv2D::GetPadDown() const { return this->primitive_->value.AsConv2D()->padDown; }
int Conv2D::GetPadLeft() const { return this->primitive_->value.AsConv2D()->padLeft; }
int Conv2D::GetPadRight() const { return this->primitive_->value.AsConv2D()->padRight; }
int Conv2D::GetDilateW() const { return this->primitive_->value.AsConv2D()->dilateW; }
int Conv2D::GetDilateH() const { return this->primitive_->value.AsConv2D()->dilateH; }
int Conv2D::GetActivationType() const { return this->primitive_->value.AsConv2D()->activationType; }

void Conv2D::SetFormat(int format) { this->primitive_->value.AsConv2D()->format = (schema::Format)format; }
void Conv2D::SetGroup(int group) { this->primitive_->value.AsConv2D()->group = group; }
void Conv2D::SetChannelIn(int channel_in) { this->primitive_->value.AsConv2D()->channelIn = channel_in; }
void Conv2D::SetChannelOut(int channel_out) { this->primitive_->value.AsConv2D()->channelOut = channel_out; }
void Conv2D::SetKernelW(int kernel_w) { this->primitive_->value.AsConv2D()->kernelW = kernel_w; }
void Conv2D::SetKernelH(int kernel_h) { this->primitive_->value.AsConv2D()->kernelH = kernel_h; }
void Conv2D::SetStrideW(int stride_w) { this->primitive_->value.AsConv2D()->strideW = stride_w; }
void Conv2D::SetStrideH(int stride_h) { this->primitive_->value.AsConv2D()->strideH = stride_h; }
void Conv2D::SetPadMode(int pad_mode) { this->primitive_->value.AsConv2D()->padMode = (schema::PadMode)pad_mode; }
void Conv2D::SetPadUp(int pad_up) { this->primitive_->value.AsConv2D()->padUp = pad_up; }
void Conv2D::SetPadDown(int pad_down) { this->primitive_->value.AsConv2D()->padDown = pad_down; }
void Conv2D::SetPadLeft(int pad_left) { this->primitive_->value.AsConv2D()->padLeft = pad_left; }
void Conv2D::SetPadRight(int pad_right) { this->primitive_->value.AsConv2D()->padRight = pad_right; }
void Conv2D::SetDilateW(int dilate_w) { this->primitive_->value.AsConv2D()->dilateW = dilate_w; }
void Conv2D::SetDilateH(int dilate_h) { this->primitive_->value.AsConv2D()->dilateH = dilate_h; }
void Conv2D::SetActivationType(int activation_type) {
  this->primitive_->value.AsConv2D()->activationType = (schema::ActivationType)activation_type;
}
template <typename T>
void ConvertConvWeight(const ParameterPtr &param_node) {
  MS_ASSERT(param_node != nullptr);
  auto param = param_node->default_param();
  auto weight = std::dynamic_pointer_cast<ParamValueLite>(param);
  MS_ASSERT(weight != nullptr);

  std::unique_ptr<T[]> buf(new (std::nothrow) T[weight->tensor_shape_size()]);

  if (buf == nullptr) {
    MS_LOG(ERROR) << "new buf failed";
    return;
  }

  size_t filter_k = weight->tensor_shape().at(0);
  size_t filter_c = weight->tensor_shape().at(1);
  size_t filter_h = weight->tensor_shape().at(2);
  size_t filter_w = weight->tensor_shape().at(3);
  T *p1Buff = nullptr;
  T *p2Buff = nullptr;
  for (size_t k = 0; k < filter_k; ++k) {
    for (size_t c = 0; c < filter_c; ++c) {
      for (size_t h = 0; h < filter_h; ++h) {
        for (size_t w = 0; w < filter_w; ++w) {
          p1Buff = reinterpret_cast<float *>(weight->tensor_addr()) +
                   ((k * filter_c * filter_h * filter_w) + (c * filter_h * filter_w) + (h * filter_w) + (w));
          p2Buff =
            buf.get() + ((c * filter_k * filter_h * filter_w) + (k * filter_h * filter_w) + (h * filter_w) + (w));
          *p2Buff = *p1Buff;
        }
      }
    }
  }

  auto ret = ::memcpy_s(weight->tensor_addr(), weight->tensor_shape_size() * sizeof(T), buf.get(),
                        weight->tensor_shape_size() * sizeof(T));
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed: " << ret;
    return;
  }

  auto abstract_base = param_node->abstract();
  MS_ASSERT(abstract_base != nullptr);
  if (utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
    utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape()[0] = filter_c;
    utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape()[1] = filter_k;
    utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape()[2] = filter_h;
    utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape()[3] = filter_w;
    weight->set_tensor_shape(
      {static_cast<int>(filter_c), static_cast<int>(filter_k), static_cast<int>(filter_h), static_cast<int>(filter_w)});
  }
  return;
}
void Conv2D::PopulaterConv2DMultiGroup(const Primitive &prim, schema::PrimitiveT *primitive, const int &group,
                                       const std::vector<AnfNodePtr> &inputs) {
  auto attr = std::make_unique<schema::DepthwiseConv2DT>();
  if (attr.get() == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return;
  }
  auto format = GetValue<std::string>(prim.GetAttr("data_format"));
  if (format == "NCHW") {
    attr->format = schema::Format::Format_NCHW;
  } else if (format == "NHWC") {
    attr->format = schema::Format::Format_NHWC;
  } else {
    attr->format = schema::Format::Format_NUM_OF_FORMAT;
  }
  auto pad_list = CastToInt(prim.GetAttr("pad_list"));
  attr->padUp = pad_list.at(0);
  attr->padDown = pad_list.at(1);
  attr->padLeft = pad_list.at(2);
  attr->padRight = pad_list.at(3);

  auto dilation = CastToInt(prim.GetAttr("dilation"));
#ifdef SUPPORT_TRAIN
  attr->dilateH = dilation.at(2);
  attr->dilateW = dilation.at(3);
#else
  attr->dilateH = dilation.at(0);
  attr->dilateW = dilation.at(1);
#endif
  auto kernel_size = CastToInt(prim.GetAttr("kernel_size"));
  attr->kernelH = kernel_size.at(0);
  attr->kernelW = (kernel_size.size() > 1) ? kernel_size.at(1) : kernel_size.at(0);

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

  int channel_mutiplier = 1;
  if (prim.GetAttr("channel_mutiplier") != nullptr) {
    channel_mutiplier = CastToInt(prim.GetAttr("channel_multiplier")).front();
  }
  attr->channelMultiplier = channel_mutiplier;

  MS_ASSERT(inputs.size() == kAnfPopulaterInputNumTwo);
  auto input_node = inputs.at(kAnfPopulaterInputNumOne);
  MS_ASSERT(input_node != nullptr);
  if (input_node->isa<Parameter>()) {
    auto param_node = input_node->cast<ParameterPtr>();
    ConvertConvWeight<float>(param_node);
    auto abstractBase = param_node->abstract();
    MS_ASSERT(abstractBase != nullptr);
    if (utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
      auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
      MS_ASSERT(abstractTensor != nullptr);
      if (utils::isa<abstract::ShapePtr>(abstractTensor->BuildShape())) {
        auto dims = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
        attr->channelIn = dims.at(kAnfPopulaterInputNumOne);
      }
    }
  } else if (input_node->isa<CNode>()) {
    // The weight of convolution is the output from the other operators which could be folded by const folding pass.
    attr->channelIn = -1;
  }

  primitive->value.type = schema::PrimitiveType_DepthwiseConv2D;
  primitive->value.value = attr.release();
}

void Conv2D::PopulaterConv2DSingleGroup(const Primitive &prim, schema::PrimitiveT *primitive, const int &group) {
  auto attr = std::make_unique<schema::Conv2DT>();
  if (attr.get() == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return;
  }
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
  attr->padUp = pad_list.at(0);
  attr->padDown = pad_list.at(1);
  attr->padLeft = pad_list.at(2);
  attr->padRight = pad_list.at(3);

  auto dilation = CastToInt(prim.GetAttr("dilation"));
  attr->dilateH = dilation.at(2);
  attr->dilateW = dilation.at(3);

  auto kernel_size = CastToInt(prim.GetAttr("kernel_size"));
  attr->kernelH = kernel_size.at(0);
  attr->kernelW = (kernel_size.size() > 1) ? kernel_size.at(1) : kernel_size.at(0);

  auto stride = CastToInt(prim.GetAttr("stride"));
  attr->strideH = stride.at(2);
  attr->strideW = stride.at(3);

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
    std::string activate_name = GetValue<std::string>(prim.GetAttr("activation_name"));
    attr->activationType = kActivationTypeMap[activate_name];
  } else {
    attr->activationType = schema::ActivationType_NO_ACTIVATION;
  }

  primitive->value.type = schema::PrimitiveType_Conv2D;
  primitive->value.value = attr.release();
}

int Conv2D::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Conv2D;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Conv2D) {
    MS_LOG(ERROR) << "primitive_ type is error:" << this->primitive_->value.type;
    return RET_ERROR;
  }
  auto groupAttr = prim.GetAttr("group");
  if (groupAttr == nullptr) {
    MS_LOG(ERROR) << "conv2d op has no group attr,please check pb model";
    return RET_NULL_PTR;
  }
  int group = CastToInt(groupAttr).front();
  if (group > 1) {
    PopulaterConv2DMultiGroup(prim, this->primitive_, group, inputs);
  } else {
    PopulaterConv2DSingleGroup(prim, this->primitive_, group);
  }

  PopulaterQuantParam(prim, inputs);
  return RET_OK;
}

#else
int Conv2D::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Conv2D();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Conv2D return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateConv2D(
    *fbb, attr->format(), attr->group(), attr->channelIn(), attr->channelOut(), attr->kernelW(), attr->kernelH(),
    attr->strideW(), attr->strideH(), attr->padMode(), attr->padUp(), attr->padDown(), attr->padLeft(),
    attr->padRight(), attr->dilateW(), attr->dilateH(), attr->hasBias(), attr->activationType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Conv2D, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int Conv2D::GetFormat() const { return this->primitive_->value_as_Conv2D()->format(); }
int Conv2D::GetGroup() const { return this->primitive_->value_as_Conv2D()->group(); }
int Conv2D::GetChannelIn() const { return this->primitive_->value_as_Conv2D()->channelIn(); }
int Conv2D::GetChannelOut() const { return this->primitive_->value_as_Conv2D()->channelOut(); }
int Conv2D::GetKernelW() const { return this->primitive_->value_as_Conv2D()->kernelW(); }
int Conv2D::GetKernelH() const { return this->primitive_->value_as_Conv2D()->kernelH(); }
int Conv2D::GetStrideW() const { return this->primitive_->value_as_Conv2D()->strideW(); }
int Conv2D::GetStrideH() const { return this->primitive_->value_as_Conv2D()->strideH(); }
int Conv2D::GetPadMode() const { return this->primitive_->value_as_Conv2D()->padMode(); }
int Conv2D::GetPadUp() const { return this->primitive_->value_as_Conv2D()->padUp(); }
int Conv2D::GetPadDown() const { return this->primitive_->value_as_Conv2D()->padDown(); }
int Conv2D::GetPadLeft() const { return this->primitive_->value_as_Conv2D()->padLeft(); }
int Conv2D::GetPadRight() const { return this->primitive_->value_as_Conv2D()->padRight(); }
int Conv2D::GetDilateW() const { return this->primitive_->value_as_Conv2D()->dilateW(); }
int Conv2D::GetDilateH() const { return this->primitive_->value_as_Conv2D()->dilateH(); }
int Conv2D::GetActivationType() const { return this->primitive_->value_as_Conv2D()->activationType(); }

PrimitiveC *Conv2DCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Conv2D>(primitive); }
Registry Conv2DRegistry(schema::PrimitiveType_Conv2D, Conv2DCreator);
#endif

void Conv2D::ConvInferShape(int input_h, int input_w, int *output_h, int *output_w) {
  MS_ASSERT(this->primitive_ != nullptr);
  int kernel_w = GetKernelW();
  int kernel_h = GetKernelH();
  int stride_w = GetStrideW();
  int stride_h = GetStrideH();
  int dilate_w = GetDilateW();
  int dilate_h = GetDilateH();

  if (GetPadMode() == schema::PadMode_SAME_UPPER) {
    *output_w = std::ceil(static_cast<float>(input_w) / static_cast<float>(stride_w));
    *output_h = std::ceil(static_cast<float>(input_h) / static_cast<float>(stride_h));
    auto pad_h_all = ((*output_h - 1) * stride_h + (kernel_h - 1) * dilate_h + 1 - input_h);
    auto pad_w_all = ((*output_w - 1) * stride_w + (kernel_w - 1) * dilate_w + 1 - input_w);
    if (pad_h_all < 0) {
      pad_u_ = pad_d_ = 0;
    } else {
      pad_u_ = pad_h_all / 2;
      pad_d_ = pad_h_all - pad_u_;
    }
    if (pad_w_all < 0) {
      pad_l_ = pad_r_ = 0;
    } else {
      pad_l_ = pad_w_all / 2;
      pad_r_ = pad_w_all - pad_l_;
    }
  } else {
    *output_w = std::ceil((static_cast<float>(input_w) + pad_l_ + pad_r_ -
                           (static_cast<float>(kernel_w) - 1) * static_cast<float>(dilate_w)) /
                          static_cast<float>(stride_w));
    *output_h = std::ceil((static_cast<float>(input_h) + pad_u_ + pad_d_ -
                           (static_cast<float>(kernel_h) - 1) * static_cast<float>(dilate_h)) /
                          static_cast<float>(stride_h));
  }
}

int Conv2D::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (inputs_.size() != 2 && inputs_.size() != 3) {
    MS_LOG(ERROR) << "Conv2d should has two or three inputs";
    return RET_ERROR;
  }
  if (outputs_.size() != 1) {
    MS_LOG(ERROR) << "Conv2d should has one outputs";
    return RET_ERROR;
  }
  auto *input_tensor = inputs_.front();
  auto *weight_tensor = inputs_.at(1);
  auto *out_tensor = outputs_.front();
  MS_ASSERT(input_tensor != nullptr);
  MS_ASSERT(out_tensor != nullptr);

  out_tensor->set_format(input_tensor->format());
  out_tensor->set_data_type(input_tensor->data_type());
  pad_l_ = GetPadLeft();
  pad_u_ = GetPadUp();
  pad_d_ = GetPadDown();
  pad_r_ = GetPadRight();

  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto in_shape = input_tensor->shape();
  int input_h = in_shape.at(1);
  int input_w = in_shape.at(2);
  int output_w = 0, output_h = 0;

  this->ConvInferShape(input_h, input_w, &output_h, &output_w);

  std::vector<int> out_shape{input_tensor->shape()};
  out_shape.at(1) = output_h > 0 ? output_h : 1;
  out_shape.at(2) = output_w > 0 ? output_w : 1;
  out_shape.at(3) = weight_tensor->shape()[0];
  out_tensor->set_shape(out_shape);

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
