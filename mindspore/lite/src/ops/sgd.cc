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
#include "src/ops/sgd.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float Sgd::GetWeightDecay() const { return this->primitive_->value.AsSgd()->weightDecay; }
float Sgd::GetDampening() const { return this->primitive_->value.AsSgd()->dampening; }
bool Sgd::GetUseNesterov() const { return this->primitive_->value.AsSgd()->useNesterov; }

int Sgd::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Sgd;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Sgd) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = std::make_unique<schema::SgdT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->weightDecay = GetValue<float>(prim.GetAttr("weight_decay"));
    attr->dampening = GetValue<float>(prim.GetAttr("dampening"));
    attr->useNesterov = GetValue<bool>(prim.GetAttr("nesterov"));

    this->primitive_->value.value = attr.release();
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else
float Sgd::GetWeightDecay() const { return this->primitive_->value_as_Sgd()->weightDecay(); }
float Sgd::GetDampening() const { return this->primitive_->value_as_Sgd()->dampening(); }
bool Sgd::GetUseNesterov() const { return this->primitive_->value_as_Sgd()->useNesterov(); }

int Sgd::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Sgd();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Sgd return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateSgd(*fbb, attr->weightDecay(), attr->dampening(), attr->useNesterov());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Sgd, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *SgdCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Sgd>(primitive); }
Registry SgdRegistry(schema::PrimitiveType_Sgd, SgdCreator);

#endif

int Sgd::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  if (6 != inputs.size()) {
    MS_LOG(ERROR) << "Sgd should have at least 6 input tensors";
    return RET_ERROR;
  }

  if (inputs.at(0)->ElementsNum() != inputs.at(1)->ElementsNum() ||
      inputs.at(0)->ElementsNum() != inputs.at(3)->ElementsNum() || inputs.at(2)->ElementsNum() != 1 ||
      inputs.at(4)->ElementsNum() != 1) {
    MS_LOG(ERROR) << "error input data size!";
    return RET_ERROR;
  }
  if (!outputs.empty()) {
    auto *out = outputs.front();
    MS_ASSERT(out != nullptr);
    out->set_data_type(inputs.at(0)->data_type());
    out->set_format(inputs.at(0)->format());
    out->set_shape({1});
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
