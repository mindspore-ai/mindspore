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

#include "src/ops/matmul.h"
#include <memory>
#include <utility>
#ifdef PRIMITIVE_WRITEABLE
#include "src/param_value_lite.h"
#endif

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
bool MatMul::GetTransposeA() const { return this->primitive_->value.AsMatMul()->transposeA; }
bool MatMul::GetTransposeB() const { return this->primitive_->value.AsMatMul()->transposeB; }

void MatMul::SetTransposeA(bool transpose_a) { this->primitive_->value.AsMatMul()->transposeA = transpose_a; }
void MatMul::SetTransposeB(bool transpose_b) { this->primitive_->value.AsMatMul()->transposeB = transpose_b; }

int MatMul::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_MatMul;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_MatMul) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::MatMulT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->transposeA = GetValue<bool>(prim.GetAttr("transpose_a"));
    attr->transposeB = GetValue<bool>(prim.GetAttr("transpose_b"));
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }

  PopulaterQuantParam(prim, inputs);
  return RET_OK;
}

#else

bool MatMul::GetTransposeA() const { return this->primitive_->value_as_MatMul()->transposeA(); }
bool MatMul::GetTransposeB() const { return this->primitive_->value_as_MatMul()->transposeB(); }

int MatMul::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_MatMul();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_MatMul return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateMatMul(*fbb, attr->broadcast(), attr->transposeA(), attr->transposeB());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_MatMul, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *MatMulCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<MatMul>(primitive); }
Registry MatMulRegistry(schema::PrimitiveType_MatMul, MatMulCreator);
#endif

int MatMul::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input0 = inputs_.front();
  MS_ASSERT(input0 != nullptr);
  auto input1 = inputs_.at(1);
  MS_ASSERT(input1 != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  output->set_data_type(input0->data_type());
  output->set_format(input0->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  std::vector<int> a_shape = input0->shape();
  std::vector<int> b_shape = input1->shape();

  if (a_shape.size() == 4 && a_shape[2] == 1 && a_shape[3] == 1) {
    a_shape.resize(2);
    input0->set_shape(a_shape);
  }

  bool del_start = false;
  bool del_end = false;
  if (a_shape.size() == 1) {
    a_shape.insert(a_shape.begin(), 1);
    input0->set_shape(a_shape);
    del_start = true;
  }
  if (b_shape.size() == 1) {
    b_shape.push_back(1);
    input1->set_shape(b_shape);
    del_end = true;
  }
  for (size_t i = 0; i < (a_shape.size() - 2) && i < (b_shape.size() - 2); ++i) {
    if (a_shape.at(a_shape.size() - 3 - i) != b_shape.at(b_shape.size() - 3 - i)) {
      MS_LOG(ERROR) << "Op MatMul's dimensions must be equal";
      return RET_INPUT_TENSOR_ERROR;
    }
  }

  if (GetTransposeA()) {
    std::swap(a_shape[a_shape.size() - 1], a_shape[a_shape.size() - 2]);
  }
  if (GetTransposeB()) {
    std::swap(b_shape[b_shape.size() - 1], b_shape[b_shape.size() - 2]);
  }
  std::vector<int> c_shape(a_shape);
  c_shape[c_shape.size() - 1] = b_shape[b_shape.size() - 1];
  if (del_start) {
    c_shape.erase(c_shape.begin());
  }
  if (del_end) {
    c_shape.pop_back();
  }
  output->set_shape(c_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
