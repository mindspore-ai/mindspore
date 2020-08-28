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

#include "src/ops/batch_norm.h"
#include <memory>
namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float BatchNorm::GetEpsilon() const { return this->primitive_->value.AsBatchNorm()->epsilon; }

void BatchNorm::SetEpsilon(float epsilon) { this->primitive_->value.AsBatchNorm()->epsilon = epsilon; }

int BatchNorm::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_FusedBatchNorm;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_FusedBatchNorm) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::FusedBatchNormT();
    attr->epsilon = GetValue<float>(prim.GetAttr("epsilon"));
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

#else
int BatchNorm::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateBatchNorm(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_BatchNorm, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
float BatchNorm::GetEpsilon() const { return this->primitive_->value_as_BatchNorm()->epsilon(); }

#endif
}  // namespace lite
}  // namespace mindspore
