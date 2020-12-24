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

#include "src/ops/conv2d_grad_input.h"
#include "src/ops/group_conv2d_grad_input.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Conv2DGradInput::GetFormat() const { return this->primitive_->value.AsConv2DGradInput()->format; }
int Conv2DGradInput::GetGroup() const { return this->primitive_->value.AsConv2DGradInput()->group; }
int Conv2DGradInput::GetChannelIn() const { return this->primitive_->value.AsConv2DGradInput()->channelIn; }
int Conv2DGradInput::GetChannelOut() const { return this->primitive_->value.AsConv2DGradInput()->channelOut; }
int Conv2DGradInput::GetKernelW() const { return this->primitive_->value.AsConv2DGradInput()->kernelW; }
int Conv2DGradInput::GetKernelH() const { return this->primitive_->value.AsConv2DGradInput()->kernelH; }
int Conv2DGradInput::GetStrideW() const { return this->primitive_->value.AsConv2DGradInput()->strideW; }
int Conv2DGradInput::GetStrideH() const { return this->primitive_->value.AsConv2DGradInput()->strideH; }
int Conv2DGradInput::GetPadMode() const { return this->primitive_->value.AsConv2DGradInput()->padMode; }
int Conv2DGradInput::GetPadUp() const { return this->primitive_->value.AsConv2DGradInput()->padUp; }
int Conv2DGradInput::GetPadDown() const { return this->primitive_->value.AsConv2DGradInput()->padDown; }
int Conv2DGradInput::GetPadLeft() const { return this->primitive_->value.AsConv2DGradInput()->padLeft; }
int Conv2DGradInput::GetPadRight() const { return this->primitive_->value.AsConv2DGradInput()->padRight; }
int Conv2DGradInput::GetDilateW() const { return this->primitive_->value.AsConv2DGradInput()->dilateW; }
int Conv2DGradInput::GetDilateH() const { return this->primitive_->value.AsConv2DGradInput()->dilateH; }
std::vector<int> Conv2DGradInput::GetInputShape() const {
  return this->primitive_->value.AsConv2DGradInput()->input_shape;
}
int Conv2DGradInput::GetActivationType() const { return this->primitive_->value.AsConv2DGradInput()->activationType; }

void Conv2DGradInput::SetFormat(int format) {
  this->primitive_->value.AsConv2DGradInput()->format = (schema::Format)format;
}
void Conv2DGradInput::SetGroup(int group) { this->primitive_->value.AsConv2DGradInput()->group = group; }
void Conv2DGradInput::SetChannelIn(int channel_in) {
  this->primitive_->value.AsConv2DGradInput()->channelIn = channel_in;
}
void Conv2DGradInput::SetChannelOut(int channel_out) {
  this->primitive_->value.AsConv2DGradInput()->channelOut = channel_out;
}
void Conv2DGradInput::SetKernelW(int kernel_w) { this->primitive_->value.AsConv2DGradInput()->kernelW = kernel_w; }
void Conv2DGradInput::SetKernelH(int kernel_h) { this->primitive_->value.AsConv2DGradInput()->kernelH = kernel_h; }
void Conv2DGradInput::SetStrideW(int stride_w) { this->primitive_->value.AsConv2DGradInput()->strideW = stride_w; }
void Conv2DGradInput::SetStrideH(int stride_h) { this->primitive_->value.AsConv2DGradInput()->strideH = stride_h; }
void Conv2DGradInput::SetPadMode(int pad_mode) {
  this->primitive_->value.AsConv2DGradInput()->padMode = (schema::PadMode)pad_mode;
}
void Conv2DGradInput::SetPadUp(int pad_up) { this->primitive_->value.AsConv2DGradInput()->padUp = pad_up; }
void Conv2DGradInput::SetPadDown(int pad_down) { this->primitive_->value.AsConv2DGradInput()->padDown = pad_down; }
void Conv2DGradInput::SetPadLeft(int pad_left) { this->primitive_->value.AsConv2DGradInput()->padLeft = pad_left; }
void Conv2DGradInput::SetPadRight(int pad_right) { this->primitive_->value.AsConv2DGradInput()->padRight = pad_right; }
void Conv2DGradInput::SetDilateW(int dilate_w) { this->primitive_->value.AsConv2DGradInput()->dilateW = dilate_w; }
void Conv2DGradInput::SetDilateH(int dilate_h) { this->primitive_->value.AsConv2DGradInput()->dilateH = dilate_h; }
void Conv2DGradInput::SetActivationType(int activation_type) {
  this->primitive_->value.AsConv2DGradInput()->activationType = (schema::ActivationType)activation_type;
}

int Conv2DGradInput::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Conv2DGradInput;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Conv2DGradInput) {
    MS_LOG(ERROR) << "primitive_ type is error:" << this->primitive_->value.type;
    return RET_ERROR;
  }

  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::Conv2DGradInputT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->group = CastToInt(prim.GetAttr("group")).front();
    if (attr->group > 1) {
      this->primitive_->value.type = schema::PrimitiveType_GroupConv2DGradInput;
    }
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
      auto input_shape = inputs[kAnfPopulaterInputNumTwo];
      MS_ASSERT(input_shape != nullptr);
      if (input_shape->isa<ValueNode>()) {
        auto valueNode = input_shape->cast<ValueNodePtr>();
        MS_ASSERT(valueNode != nullptr);
        auto value = valueNode->value();
        MS_ASSERT(value != nullptr);
        if (value->isa<ValueTuple>()) {
          auto valTuplPtr = dyn_cast<ValueTuple>(value);
          MS_ASSERT(valTuplPtr != nullptr);
          const int nchw2nhwc[] = {0, 3, 1, 2};
          attr->input_shape.resize(valTuplPtr->size());
          for (size_t i = 0; i < valTuplPtr->size(); i++) {
            auto elem = (*valTuplPtr)[i];
            MS_ASSERT(elem != nullptr);
            attr->input_shape[nchw2nhwc[i]] = CastToInt(elem).front();
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
int Conv2DGradInput::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Conv2DGradInput();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Conv2DGradInput return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> input_shape;
  if (attr->input_shape() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->input_shape()->size()); i++) {
      input_shape.push_back(attr->input_shape()->data()[i]);
    }
  }
  auto val_offset = schema::CreateConv2DGradInputDirect(
    *fbb, attr->format(), attr->group(), attr->channelIn(), attr->channelOut(), attr->kernelW(), attr->kernelH(),
    attr->strideW(), attr->strideH(), attr->padMode(), attr->padUp(), attr->padDown(), attr->padLeft(),
    attr->padRight(), attr->dilateW(), attr->dilateH(), attr->hasBias(), &input_shape, attr->activationType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Conv2DGradInput, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int Conv2DGradInput::GetFormat() const { return this->primitive_->value_as_Conv2DGradInput()->format(); }
int Conv2DGradInput::GetGroup() const { return this->primitive_->value_as_Conv2DGradInput()->group(); }
int Conv2DGradInput::GetChannelIn() const { return this->primitive_->value_as_Conv2DGradInput()->channelIn(); }
int Conv2DGradInput::GetChannelOut() const { return this->primitive_->value_as_Conv2DGradInput()->channelOut(); }
int Conv2DGradInput::GetKernelW() const { return this->primitive_->value_as_Conv2DGradInput()->kernelW(); }
int Conv2DGradInput::GetKernelH() const { return this->primitive_->value_as_Conv2DGradInput()->kernelH(); }
int Conv2DGradInput::GetStrideW() const { return this->primitive_->value_as_Conv2DGradInput()->strideW(); }
int Conv2DGradInput::GetStrideH() const { return this->primitive_->value_as_Conv2DGradInput()->strideH(); }
int Conv2DGradInput::GetPadMode() const { return this->primitive_->value_as_Conv2DGradInput()->padMode(); }
int Conv2DGradInput::GetPadUp() const { return this->primitive_->value_as_Conv2DGradInput()->padUp(); }
int Conv2DGradInput::GetPadDown() const { return this->primitive_->value_as_Conv2DGradInput()->padDown(); }
int Conv2DGradInput::GetPadLeft() const { return this->primitive_->value_as_Conv2DGradInput()->padLeft(); }
int Conv2DGradInput::GetPadRight() const { return this->primitive_->value_as_Conv2DGradInput()->padRight(); }
int Conv2DGradInput::GetDilateW() const { return this->primitive_->value_as_Conv2DGradInput()->dilateW(); }
int Conv2DGradInput::GetDilateH() const { return this->primitive_->value_as_Conv2DGradInput()->dilateH(); }
std::vector<int> Conv2DGradInput::GetInputShape() const {
  auto fb_vector = this->primitive_->value_as_Conv2DGradInput()->input_shape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Conv2DGradInput::GetActivationType() const {
  return this->primitive_->value_as_Conv2DGradInput()->activationType();
}

PrimitiveC *Conv2DGradInputCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<Conv2DGradInput>(primitive);
}
Registry Conv2DGradInputRegistry(schema::PrimitiveType_Conv2DGradInput, Conv2DGradInputCreator);
#endif

int Conv2DGradInput::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  if (inputs.size() < 2) {
    MS_LOG(ERROR) << "Conv2d Grad Input should be at least two input";
    return RET_ERROR;
  }
  if (outputs.size() != 1) {
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
