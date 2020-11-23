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
#include "src/ops/apply_momentum.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float ApplyMomentum::GetGradientScale() const { return this->primitive_->value.AsApplyMomentum()->gradientScale; }
bool ApplyMomentum::GetUseNesterov() const { return this->primitive_->value.AsApplyMomentum()->useNesterov; }

int ApplyMomentum::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_ApplyMomentum;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_ApplyMomentum) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = std::make_unique<schema::ApplyMomentumT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->gradientScale = GetValue<float>(prim.GetAttr("gradient_scale"));
    attr->useNesterov = GetValue<bool>(prim.GetAttr("use_nesterov"));

    this->primitive_->value.value = attr.release();
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else
float ApplyMomentum::GetGradientScale() const { return this->primitive_->value_as_ApplyMomentum()->gradientScale(); }
bool ApplyMomentum::GetUseNesterov() const { return this->primitive_->value_as_ApplyMomentum()->useNesterov(); }

int ApplyMomentum::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_ApplyMomentum();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_ApplyMomentum return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateApplyMomentum(*fbb, attr->gradientScale(), attr->useNesterov());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_ApplyMomentum, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *ApplyMomentumCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<ApplyMomentum>(primitive);
}
Registry ApplyMomentumRegistry(schema::PrimitiveType_ApplyMomentum, ApplyMomentumCreator);
#endif

int ApplyMomentum::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  if (inputs.size() != 5) {
    MS_LOG(ERROR) << "ApplyMomentum should have at least 5 input tensors";
    return RET_ERROR;
  }

  if (inputs[0]->ElementsNum() != inputs[1]->ElementsNum() || inputs[0]->ElementsNum() != inputs[3]->ElementsNum() ||
      inputs[2]->ElementsNum() != 1 || inputs[4]->ElementsNum() != 1) {
    MS_LOG(ERROR) << "error input data size!";
    return RET_ERROR;
  }
  if (!outputs.empty()) {
    auto *out = outputs.front();
    MS_ASSERT(out != nullptr);
    out->set_data_type(inputs[0]->data_type());
    out->set_format(inputs[0]->format());
    out->set_shape({1});
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
