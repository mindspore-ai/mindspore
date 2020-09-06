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

#include "src/ops/bn_grad.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float BNGrad::GetEps() const { return this->primitive_->value.AsBNGrad()->eps; }
float BNGrad::GetMomentum() const { return this->primitive_->value.AsBNGrad()->momentum; }

void BNGrad::SetEps(float eps) { this->primitive_->value.AsBNGrad()->eps = eps; }
void BNGrad::SetMomentum(float momentum) { this->primitive_->value.AsBNGrad()->momentum = momentum; }

#else
int BNGrad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_BNGrad();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_BNGradInput return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateBNGrad(*fbb, attr->eps(), attr->momentum());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_BNGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
float BNGrad::GetEps() const { return this->primitive_->value_as_BNGrad()->eps(); }
float BNGrad::GetMomentum() const { return this->primitive_->value_as_BNGrad()->momentum(); }

#endif
}  // namespace lite
}  // namespace mindspore
