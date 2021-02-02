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

#include "src/ops/pooling_grad.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int PoolingGrad::GetFormat() const { return this->primitive_->value.AsPoolingGrad()->format; }
int PoolingGrad::GetPoolingMode() const { return this->primitive_->value.AsPoolingGrad()->poolingMode; }
bool PoolingGrad::GetGlobal() const { return this->primitive_->value.AsPoolingGrad()->global; }
int PoolingGrad::GetWindowW() const { return this->primitive_->value.AsPoolingGrad()->windowW; }
int PoolingGrad::GetWindowH() const { return this->primitive_->value.AsPoolingGrad()->windowH; }
int PoolingGrad::GetStrideW() const { return this->primitive_->value.AsPoolingGrad()->strideW; }
int PoolingGrad::GetStrideH() const { return this->primitive_->value.AsPoolingGrad()->strideH; }
int PoolingGrad::GetPadMode() const { return this->primitive_->value.AsPoolingGrad()->padMode; }
int PoolingGrad::GetPadUp() const { return this->primitive_->value.AsPoolingGrad()->padUp; }
int PoolingGrad::GetPadDown() const { return this->primitive_->value.AsPoolingGrad()->padDown; }
int PoolingGrad::GetPadLeft() const { return this->primitive_->value.AsPoolingGrad()->padLeft; }
int PoolingGrad::GetPadRight() const { return this->primitive_->value.AsPoolingGrad()->padRight; }
int PoolingGrad::GetRoundMode() const { return this->primitive_->value.AsPoolingGrad()->roundMode; }

void PoolingGrad::SetFormat(int format) { this->primitive_->value.AsPoolingGrad()->format = (schema::Format)format; }
void PoolingGrad::SetPoolingMode(int pooling_mode) {
  this->primitive_->value.AsPoolingGrad()->poolingMode = (schema::PoolMode)pooling_mode;
}
void PoolingGrad::SetGlobal(bool global) { this->primitive_->value.AsPoolingGrad()->global = global; }
void PoolingGrad::SetWindowW(int window_w) { this->primitive_->value.AsPoolingGrad()->windowW = window_w; }
void PoolingGrad::SetWindowH(int window_h) { this->primitive_->value.AsPoolingGrad()->windowH = window_h; }
void PoolingGrad::SetStrideW(int stride_w) { this->primitive_->value.AsPoolingGrad()->strideW = stride_w; }
void PoolingGrad::SetStrideH(int stride_h) { this->primitive_->value.AsPoolingGrad()->strideH = stride_h; }
void PoolingGrad::SetPadMode(int pad_mode) {
  this->primitive_->value.AsPoolingGrad()->padMode = (schema::PadMode)pad_mode;
}
void PoolingGrad::SetPadUp(int pad_up) { this->primitive_->value.AsPoolingGrad()->padUp = pad_up; }
void PoolingGrad::SetPadDown(int pad_down) { this->primitive_->value.AsPoolingGrad()->padDown = pad_down; }
void PoolingGrad::SetPadLeft(int pad_left) { this->primitive_->value.AsPoolingGrad()->padLeft = pad_left; }
void PoolingGrad::SetPadRight(int pad_right) { this->primitive_->value.AsPoolingGrad()->padRight = pad_right; }
void PoolingGrad::SetRoundMode(int round_mode) {
  this->primitive_->value.AsPoolingGrad()->roundMode = (schema::RoundMode)round_mode;
}
int PoolingGrad::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_PoolingGrad;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_PoolingGrad) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::PoolingGradT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }

    auto format = GetValue<std::string>(prim.GetAttr("data_format"));
    if (format == "NCHW") {
      attr->format = schema::Format_NCHW;
    } else if (format == "NHWC") {
      attr->format = schema::Format_NHWC;
    } else {
      attr->format = schema::Format_NUM_OF_FORMAT;
    }

    if (prim.instance_name() == "MaxPoolGrad") {
      attr->poolingMode = schema::PoolMode_MAX_POOLING;
    } else if (prim.instance_name() == "AvgPoolGrad") {
      attr->poolingMode = schema::PoolMode_MEAN_POOLING;
    } else if (prim.instance_name() == "AvgPoolGradGpu") {
      attr->poolingMode = schema::PoolMode_MEAN_POOLING;
    } else if (prim.instance_name() == "AvgPoolGradCpu") {
      attr->poolingMode = schema::PoolMode_MEAN_POOLING;
    } else {
      attr->poolingMode = schema::PoolMode_MAX_POOLING;
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

int PoolingGrad::GetFormat() const { return this->primitive_->value_as_PoolingGrad()->format(); }
int PoolingGrad::GetPoolingMode() const { return this->primitive_->value_as_PoolingGrad()->poolingMode(); }
bool PoolingGrad::GetGlobal() const { return this->primitive_->value_as_PoolingGrad()->global(); }
int PoolingGrad::GetWindowW() const { return this->primitive_->value_as_PoolingGrad()->windowW(); }
int PoolingGrad::GetWindowH() const { return this->primitive_->value_as_PoolingGrad()->windowH(); }
int PoolingGrad::GetStrideW() const { return this->primitive_->value_as_PoolingGrad()->strideW(); }
int PoolingGrad::GetStrideH() const { return this->primitive_->value_as_PoolingGrad()->strideH(); }
int PoolingGrad::GetPadMode() const { return this->primitive_->value_as_PoolingGrad()->padMode(); }
int PoolingGrad::GetPadUp() const { return this->primitive_->value_as_PoolingGrad()->padUp(); }
int PoolingGrad::GetPadDown() const { return this->primitive_->value_as_PoolingGrad()->padDown(); }
int PoolingGrad::GetPadLeft() const { return this->primitive_->value_as_PoolingGrad()->padLeft(); }
int PoolingGrad::GetPadRight() const { return this->primitive_->value_as_PoolingGrad()->padRight(); }
int PoolingGrad::GetRoundMode() const { return this->primitive_->value_as_PoolingGrad()->roundMode(); }

int PoolingGrad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_PoolingGrad();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_PoolingGrad return nullptr";
    return RET_ERROR;
  }
  auto val_offset =
    schema::CreatePoolingGrad(*fbb, attr->format(), attr->poolingMode(), attr->global(), attr->windowW(),
                              attr->windowH(), attr->strideW(), attr->strideH(), attr->padMode(), attr->padUp(),
                              attr->padDown(), attr->padLeft(), attr->padRight(), attr->roundMode());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_PoolingGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *PoolingGradCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<PoolingGrad>(primitive);
}
Registry PoolingGradRegistry(schema::PrimitiveType_PoolingGrad, PoolingGradCreator);
#endif

int PoolingGrad::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (3 != inputs_.size()) {
    MS_LOG(ERROR) << "Pooling Grad Filter should have 3 inputs";
    return RET_ERROR;
  }
  if (1 != outputs_.size()) {
    MS_LOG(ERROR) << "Pooling Grad Filter should have one output";
    return RET_ERROR;
  }

  auto input = inputs_.at(0);
  MS_ASSERT(input != nullptr);
  int input_h = input->shape().at(1);
  int input_w = input->shape().at(2);

  auto window_h = GetWindowH();
  auto window_w = GetWindowW();
  if (GetGlobal()) {
    window_h = input_h;
    window_w = input_w;
  }

  pad_l_ = GetPadLeft();
  pad_u_ = GetPadUp();
  pad_d_ = GetPadDown();
  pad_r_ = GetPadRight();
  if (GetPadMode() == schema::PadMode_SAME_UPPER) {
    int output_w = std::ceil(static_cast<float>(input_w) / static_cast<float>(GetStrideW()));
    int output_h = std::ceil(static_cast<float>(input_h) / static_cast<float>(GetStrideH()));
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
  }
  auto grad_output = outputs_.at(0);
  auto output_shape = input->shape();
  grad_output->set_shape(output_shape);
  grad_output->set_data_type(input->data_type());
  grad_output->set_format(input->format());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
