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
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
bool BatchMatMul::GetTransposeA() const { return this->primitive_->value.AsBatchMatMul()->transpose_a; }

bool BatchMatMul::GetTransposeB() const { return this->primitive_->value.AsBatchMatMul()->transpose_b; }

void BatchMatMul::SetTransposeA(bool transpose_a) {
  this->primitive_->value.AsBatchMatMul()->transpose_a = transpose_a;
}

void BatchMatMul::SetTransposeB(bool transpose_b) {
  this->primitive_->value.AsBatchMatMul()->transpose_b = transpose_b;
}
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
    attr->transpose_a = GetValue<bool>(prim.GetAttr("transpose_a"));
    attr->transpose_b = GetValue<bool>(prim.GetAttr("transpose_b"));
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}
#else
bool BatchMatMul::GetTransposeA() const { return this->primitive_->value_as_BatchMatMul()->transpose_a(); }
bool BatchMatMul::GetTransposeB() const { return this->primitive_->value_as_BatchMatMul()->transpose_b(); }
int BatchMatMul::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_BatchMatMul();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Add return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateBatchMatMul(*fbb, attr->transpose_a(), attr->transpose_b());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_BatchMatMul, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *BatchMatMulCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<BatchMatMul>(primitive);
}
Registry BatchMatMulRegistry(schema::PrimitiveType_BatchMatMul, BatchMatMulCreator);
#endif
}  // namespace lite
}  // namespace mindspore
