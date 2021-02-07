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
#include "src/ops/batch_matmul.h"
#include <memory>
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
bool BatchMatMul::GetAdjX() const { return this->primitive_->value.AsBatchMatMul()->adj_x; }

void BatchMatMul::SetAdjX(bool adj_x) { this->primitive_->value.AsBatchMatMul()->adj_x = adj_x; }

bool BatchMatMul::GetAdjY() const { return this->primitive_->value.AsBatchMatMul()->adj_y; }

void BatchMatMul::SetAdjY(bool adj_y) { this->primitive_->value.AsBatchMatMul()->adj_y = adj_y; }

int BatchMatMul::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_BatchMatMul;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_BatchMatMul) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::BatchMatMulT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new FusedBatchMatMulT failed";
      delete this->primitive_;
      this->primitive_ = nullptr;
      return RET_ERROR;
    }
    attr->adj_x = GetValue<bool>(prim.GetAttr("adj_x"));
    attr->adj_y = GetValue<bool>(prim.GetAttr("adj_y"));
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}

#else
int BatchMatMul::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateBatchMatMul(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_BatchMatMul, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
bool BatchMatMul::GetAdjX() const { return this->primitive_->value_as_BatchMatMul()->adj_x(); }

bool BatchMatMul::GetAdjY() const { return this->primitive_->value_as_BatchMatMul()->adj_y(); }

PrimitiveC *BatchMatMulCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<BatchMatMul>(primitive);
}
Registry BatchMatMulRegistry(schema::PrimitiveType_BatchMatMul, BatchMatMulCreator);
#endif

}  // namespace lite
}  // namespace mindspore
