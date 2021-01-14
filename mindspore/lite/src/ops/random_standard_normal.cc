/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/ops/random_standard_normal.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int RandomStandardNormal::GetSeed() const { return this->primitive_->value.AsRandomStandardNormal()->seed; }

int RandomStandardNormal::GetSeed2() const { return this->primitive_->value.AsRandomStandardNormal()->seed2; }

int RandomStandardNormal::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_RandomStandardNormal;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_RandomStandardNormal) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::RandomStandardNormalT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
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
int RandomStandardNormal::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_RandomStandardNormal();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_RandomStandardNormal return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateRandomStandardNormal(*fbb, attr->seed(), attr->seed2());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_RandomStandardNormal, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

int RandomStandardNormal::GetSeed() const { return this->primitive_->value_as_RandomStandardNormal()->seed(); }

int RandomStandardNormal::GetSeed2() const { return this->primitive_->value_as_RandomStandardNormal()->seed2(); }

PrimitiveC *RandomStandardNormalCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<RandomStandardNormal>(primitive);
}
Registry RandomStandardNormalRegistry(schema::PrimitiveType_RandomStandardNormal, RandomStandardNormalCreator);
#endif

int RandomStandardNormal::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto input_data = static_cast<int32_t *>(inputs_[0]->data_c());
  if (input_data == nullptr) {
    return RET_INFER_INVALID;
  }
  auto input_num = inputs_[0]->ElementsNum();
  std::vector<int> output_shape = {};
  for (int i = 0; i < input_num; i++) {
    output_shape.push_back(input_data[i]);
  }
  outputs_[0]->set_shape(output_shape);
  outputs_[0]->set_data_type(kNumberTypeFloat32);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
