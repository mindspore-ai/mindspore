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

#include "src/ops/power.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float Power::GetPower() const { return this->primitive_->value.AsPower()->power; }
float Power::GetScale() const { return this->primitive_->value.AsPower()->scale; }
float Power::GetShift() const { return this->primitive_->value.AsPower()->shift; }

void Power::SetPower(float power) { this->primitive_->value.AsPower()->power = power; }
void Power::SetScale(float scale) { this->primitive_->value.AsPower()->scale = scale; }
void Power::SetShift(float shift) { this->primitive_->value.AsPower()->shift = shift; }

int Power::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Power;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Power) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::PowerT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      delete this->primitive_;
      this->primitive_ = nullptr;
      return RET_ERROR;
    }

    if (prim.GetAttr("scale") == nullptr) {
      MS_LOG(INFO) << "Power's attr scale is set to default";
      attr->scale = 1.0f;
    } else {
      attr->scale = GetValue<float>(prim.GetAttr("scale"));
    }
    if (prim.GetAttr("power") == nullptr) {
      MS_LOG(INFO) << "Power's attr power is set to default";
      attr->power = 1.0f;
    } else {
      attr->power = GetValue<float>(prim.GetAttr("power"));
    }
    if (prim.GetAttr("shift") == nullptr) {
      MS_LOG(INFO) << "Power's attr shift is set to default";
      attr->shift = 0;
    } else {
      attr->shift = GetValue<float>(prim.GetAttr("shift"));
    }
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}

#else

float Power::GetPower() const { return this->primitive_->value_as_Power()->power(); }
float Power::GetScale() const { return this->primitive_->value_as_Power()->scale(); }
float Power::GetShift() const { return this->primitive_->value_as_Power()->shift(); }
int Power::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Power();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Power return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreatePower(*fbb, attr->power(), attr->scale(), attr->shift());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Power, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *PowerCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Power>(primitive); }
Registry PowerRegistry(schema::PrimitiveType_Power, PowerCreator);
#endif

int Power::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto x_tensor = inputs.at(0);
  MS_ASSERT(x_tensor != nullptr);
  Tensor *exp_tensor = nullptr;
  if (inputs.size() == 2) {
    exp_tensor = inputs.at(1);
    MS_ASSERT(exp_tensor != nullptr);
  }
  auto output_tensor = outputs.at(0);
  MS_ASSERT(output_tensor != nullptr);
  output_tensor->set_data_type(x_tensor->data_type());
  output_tensor->set_format(x_tensor->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  if (exp_tensor != nullptr) {
    if ((exp_tensor->shape().size() > 1 && exp_tensor->shape() != x_tensor->shape()) ||
        (exp_tensor->shape().size() == 1 && exp_tensor->shape().at(0) != 1) ||
        exp_tensor->data_type() != x_tensor->data_type()) {
      MS_LOG(ERROR) << "Power inputs shape or type is not equal!";
      return RET_INPUT_TENSOR_ERROR;
    }
  }

  output_tensor->set_shape(x_tensor->shape());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
