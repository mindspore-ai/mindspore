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

#include "src/ops/pooling.h"
#include <memory>
#include <string>
#include <vector>

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {

#ifdef PRIMITIVE_WRITEABLE
int Pooling::GetFormat() const { return this->primitive_->value.AsPooling()->format; }
int Pooling::GetPoolingMode() const { return this->primitive_->value.AsPooling()->poolingMode; }
bool Pooling::GetGlobal() const { return this->primitive_->value.AsPooling()->global; }
int Pooling::GetWindowW() const { return this->primitive_->value.AsPooling()->windowW; }
int Pooling::GetWindowH() const { return this->primitive_->value.AsPooling()->windowH; }
int Pooling::GetStrideW() const { return this->primitive_->value.AsPooling()->strideW; }
int Pooling::GetStrideH() const { return this->primitive_->value.AsPooling()->strideH; }
int Pooling::GetPadMode() const { return this->primitive_->value.AsPooling()->padMode; }
int Pooling::GetPadUp() const { return this->primitive_->value.AsPooling()->padUp; }
int Pooling::GetPadDown() const { return this->primitive_->value.AsPooling()->padDown; }
int Pooling::GetPadLeft() const { return this->primitive_->value.AsPooling()->padLeft; }
int Pooling::GetPadRight() const { return this->primitive_->value.AsPooling()->padRight; }
int Pooling::GetRoundMode() const { return this->primitive_->value.AsPooling()->roundMode; }
int Pooling::GetActivationType() const { return this->primitive_->value.AsPooling()->activationType; }
int Pooling::GetAvgMode() const { return this->primitive_->value.AsPooling()->avgMode; }

void Pooling::SetFormat(int format) { this->primitive_->value.AsPooling()->format = (schema::Format)format; }
void Pooling::SetPoolingMode(int pooling_mode) {
  this->primitive_->value.AsPooling()->poolingMode = (schema::PoolMode)pooling_mode;
}
void Pooling::SetGlobal(bool global) { this->primitive_->value.AsPooling()->global = global; }
void Pooling::SetWindowW(int window_w) { this->primitive_->value.AsPooling()->windowW = window_w; }
void Pooling::SetWindowH(int window_h) { this->primitive_->value.AsPooling()->windowH = window_h; }
void Pooling::SetStrideW(int stride_w) { this->primitive_->value.AsPooling()->strideW = stride_w; }
void Pooling::SetStrideH(int stride_h) { this->primitive_->value.AsPooling()->strideH = stride_h; }
void Pooling::SetPadMode(int pad_mode) { this->primitive_->value.AsPooling()->padMode = (schema::PadMode)pad_mode; }
void Pooling::SetPadUp(int pad_up) { this->primitive_->value.AsPooling()->padUp = pad_up; }
void Pooling::SetPadDown(int pad_down) { this->primitive_->value.AsPooling()->padDown = pad_down; }
void Pooling::SetPadLeft(int pad_left) { this->primitive_->value.AsPooling()->padLeft = pad_left; }
void Pooling::SetPadRight(int pad_right) { this->primitive_->value.AsPooling()->padRight = pad_right; }
void Pooling::SetRoundMode(int round_mode) {
  this->primitive_->value.AsPooling()->roundMode = (schema::RoundMode)round_mode;
}
void Pooling::SetActivationType(int activation_type) {
  this->primitive_->value.AsPooling()->activationType = (schema::ActivationType)activation_type;
}
void Pooling::SetAvgMode(int avg_mode) { this->primitive_->value.AsPooling()->avgMode = avg_mode; }

int Pooling::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Pooling;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Pooling) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::PoolingT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    if (prim.instance_name() == "MaxPool") {
      attr->poolingMode = schema::PoolMode_MAX_POOLING;
    } else if (prim.instance_name() == "MeanPool" || prim.instance_name() == "AvgPool") {
      attr->poolingMode = schema::PoolMode_MEAN_POOLING;
    }

    auto format = GetValue<std::string>(prim.GetAttr("data_format"));
    if (format == "NCHW") {
      attr->format = schema::Format::Format_NCHW;
    } else if (format == "NHWC") {
      attr->format = schema::Format::Format_NHWC;
    } else {
      attr->format = schema::Format::Format_NUM_OF_FORMAT;
    }

    auto pad_mode = GetValue<std::string>(prim.GetAttr("padding"));
    if (pad_mode == "VALID") {
      attr->padMode = schema::PadMode_VALID;
    } else if (pad_mode == "SAME") {
      attr->padMode = schema::PadMode_SAME_UPPER;
    } else {
      attr->padMode = schema::PadMode_NOTSET;
    }

    auto kernel_size = CastToInt(prim.GetAttr("ksize"));
    attr->windowH = kernel_size.at(2);
    attr->windowW = kernel_size.at(3);

    auto stride = CastToInt(prim.GetAttr("strides"));
    attr->strideH = stride.at(2);
    attr->strideW = stride.at(3);
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

#else

int Pooling::GetFormat() const { return this->primitive_->value_as_Pooling()->format(); }
int Pooling::GetPoolingMode() const { return this->primitive_->value_as_Pooling()->poolingMode(); }
bool Pooling::GetGlobal() const { return this->primitive_->value_as_Pooling()->global(); }
int Pooling::GetWindowW() const { return this->primitive_->value_as_Pooling()->windowW(); }
int Pooling::GetWindowH() const { return this->primitive_->value_as_Pooling()->windowH(); }
int Pooling::GetStrideW() const { return this->primitive_->value_as_Pooling()->strideW(); }
int Pooling::GetStrideH() const { return this->primitive_->value_as_Pooling()->strideH(); }
int Pooling::GetPadMode() const { return this->primitive_->value_as_Pooling()->padMode(); }
int Pooling::GetPadUp() const { return this->primitive_->value_as_Pooling()->padUp(); }
int Pooling::GetPadDown() const { return this->primitive_->value_as_Pooling()->padDown(); }
int Pooling::GetPadLeft() const { return this->primitive_->value_as_Pooling()->padLeft(); }
int Pooling::GetPadRight() const { return this->primitive_->value_as_Pooling()->padRight(); }
int Pooling::GetRoundMode() const { return this->primitive_->value_as_Pooling()->roundMode(); }
int Pooling::GetActivationType() const { return this->primitive_->value_as_Pooling()->activationType(); }
int Pooling::GetAvgMode() const { return this->primitive_->value_as_Pooling()->avgMode(); }

int Pooling::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Pooling();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Pooling return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreatePooling(*fbb, attr->format(), attr->poolingMode(), attr->global(), attr->windowW(),
                                          attr->windowH(), attr->strideW(), attr->strideH(), attr->padMode(),
                                          attr->padUp(), attr->padDown(), attr->padLeft(), attr->padRight(),
                                          attr->roundMode(), attr->activationType(), attr->avgMode());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Pooling, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *PoolingCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Pooling>(primitive); }
Registry PoolingRegistry(schema::PrimitiveType_Pooling, PoolingCreator);

#endif

int Pooling::PadUp() const { return this->pad_u_; }
int Pooling::PadDown() const { return this->pad_d_; }
int Pooling::PadLeft() const { return this->pad_l_; }
int Pooling::PadRight() const { return this->pad_r_; }

int Pooling::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(input->data_type());
  output->set_format(schema::Format::Format_NHWC);
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  int input_h = input->shape().at(1);
  int input_w = input->shape().at(2);

  auto window_h = GetWindowH();
  auto window_w = GetWindowW();
  if (GetGlobal()) {
    window_h = input_h;
    window_w = input_w;
  }
  int output_h = 0;
  int output_w = 0;
  pad_l_ = GetPadLeft();
  pad_u_ = GetPadUp();
  pad_d_ = GetPadDown();
  pad_r_ = GetPadRight();
  if (GetPadMode() == schema::PadMode_SAME_UPPER) {
    output_w = std::ceil(static_cast<float>(input_w) / static_cast<float>(GetStrideW()));
    output_h = std::ceil(static_cast<float>(input_h) / static_cast<float>(GetStrideH()));
    auto pad_h_all = ((output_h - 1) * GetStrideH() + (window_h - 1) + 1 - input_h);
    auto pad_w_all = ((output_w - 1) * GetStrideW() + (window_w - 1) + 1 - input_w);
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
    auto round_mode = (schema::RoundMode)GetRoundMode();
    if (round_mode == schema::RoundMode_FLOOR) {
      output_h = std::floor(static_cast<float>(input_h + pad_u_ + pad_d_ - window_h) / GetStrideH()) + 1;
      output_w = std::floor(static_cast<float>(input_w + pad_l_ + pad_r_ - window_w) / GetStrideW()) + 1;
    } else if (round_mode == schema::RoundMode_CEIL) {
      output_h = std::ceil(static_cast<float>(input_h + pad_u_ + pad_d_ - window_h) / GetStrideH()) + 1;
      output_w = std::ceil(static_cast<float>(input_w + pad_l_ + pad_r_ - window_w) / GetStrideW()) + 1;
    } else {
      MS_LOG(ERROR) << "unsupported round mode.";
    }
  }
  auto input_shape = input->shape();
  input_shape.at(1) = output_h > 0 ? output_h : 1;
  input_shape.at(2) = output_w > 0 ? output_w : 1;
  output->set_shape(input_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
