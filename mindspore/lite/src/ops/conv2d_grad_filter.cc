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
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

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
std::vector<int> Conv2DGradFilter::GetFilterShape() const {
  return this->primitive_->value.AsConv2DGradFilter()->filter_shape;
}
void Conv2DGradFilter::SetActivationType(int activation_type) {
  this->primitive_->value.AsConv2DGradFilter()->activationType = (schema::ActivationType)activation_type;
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

  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::Conv2DGradFilterT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->group = CastToInt(prim.GetAttr("group")).front();
    auto format = GetValue<std::string>(prim.GetAttr("data_format"));
    if (format == "NCHW") {
      attr->format = schema::Format_NCHW;
    } else if (format == "NHWC") {
      attr->format = schema::Format_NHWC;
    } else {
      attr->format = schema::Format_NUM_OF_FORMAT;
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
    attr->strideH = stride.at(0);
    attr->strideW = stride.at(1);

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

    if (inputs.size() >= kAnfPopulaterInputNumThree) {
      auto filter_shape = inputs[kAnfPopulaterInputNumTwo];
      MS_ASSERT(filter_shape != nullptr);
      if (filter_shape->isa<ValueNode>()) {
        auto valueNode = filter_shape->cast<ValueNodePtr>();
        MS_ASSERT(valueNode != nullptr);
        auto value = valueNode->value();
        MS_ASSERT(value != nullptr);
        if (value->isa<ValueTuple>()) {
          auto valTuplPtr = dyn_cast<ValueTuple>(value);
          MS_ASSERT(valTuplPtr != nullptr);
          const int nchw2nhwc[] = {0, 3, 1, 2};
          attr->filter_shape.resize(valTuplPtr->size());
          for (size_t i = 0; i < valTuplPtr->size(); i++) {
            auto elem = (*valTuplPtr)[i];
            MS_ASSERT(elem != nullptr);
            attr->filter_shape[nchw2nhwc[i]] = CastToInt(elem).front();
          }
        }
      }
    }

    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
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
  std::vector<int32_t> filter_shape;
  if (attr->filter_shape() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->filter_shape()->size()); i++) {
      filter_shape.push_back(attr->filter_shape()->data()[i]);
    }
  }
  auto val_offset = schema::CreateConv2DGradFilterDirect(
    *fbb, attr->format(), attr->group(), attr->channelIn(), attr->channelOut(), attr->kernelW(), attr->kernelH(),
    attr->strideW(), attr->strideH(), attr->padMode(), attr->padUp(), attr->padDown(), attr->padLeft(),
    attr->padRight(), attr->dilateW(), attr->dilateH(), attr->hasBias(), &filter_shape, attr->activationType());
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
std::vector<int> Conv2DGradFilter::GetFilterShape() const {
  auto fb_vector = this->primitive_->value_as_Conv2DGradFilter()->filter_shape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Conv2DGradFilter::GetActivationType() const {
  return this->primitive_->value_as_Conv2DGradFilter()->activationType();
}

PrimitiveC *Conv2DGradFilterCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<Conv2DGradFilter>(primitive);
}
Registry conv2DGradFilterRegistry(schema::PrimitiveType_Conv2DGradFilter, Conv2DGradFilterCreator);
#endif

int Conv2DGradFilter::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  if (inputs.size() < 2) {
    MS_LOG(ERROR) << "Conv2d Grad Filter should be at least two input, but it got " << inputs.size();
    return RET_ERROR;
  }
  if (outputs.size() != 1) {
    MS_LOG(ERROR) << "Conv2d Grad Filter should have one output but it got " << outputs.size();
    return RET_ERROR;
  }

  auto *in0 = inputs.at(0);
  MS_ASSERT(in0 != nullptr);

  auto *out = outputs.at(0);
  MS_ASSERT(out != nullptr);

  out->set_shape(GetFilterShape());
  out->set_data_type(in0->data_type());
  out->set_format(in0->format());

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
