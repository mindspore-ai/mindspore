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

#include "src/ops/sparse_to_dense.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> SparseToDense::GetOutputShape() const {
  return this->primitive_->value.AsSparseToDense()->outputShape;
}
std::vector<int> SparseToDense::GetSparseValue() const {
  return this->primitive_->value.AsSparseToDense()->sparseValue;
}
std::vector<int> SparseToDense::GetDefaultValue() const {
  return this->primitive_->value.AsSparseToDense()->defaultValue;
}
bool SparseToDense::GetValidateIndices() const { return this->primitive_->value.AsSparseToDense()->validateIndices; }

void SparseToDense::SetOutputShape(const std::vector<int> &output_shape) {
  this->primitive_->value.AsSparseToDense()->outputShape = output_shape;
}
void SparseToDense::SetSparseValue(const std::vector<int> &sparse_value) {
  this->primitive_->value.AsSparseToDense()->sparseValue = sparse_value;
}
void SparseToDense::SetDefaultValue(const std::vector<int> &default_value) {
  this->primitive_->value.AsSparseToDense()->defaultValue = default_value;
}
void SparseToDense::SetValidateIndices(bool validate_indices) {
  this->primitive_->value.AsSparseToDense()->validateIndices = validate_indices;
}

#else

std::vector<int> SparseToDense::GetOutputShape() const {
  auto fb_vector = this->primitive_->value_as_SparseToDense()->outputShape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> SparseToDense::GetSparseValue() const {
  auto fb_vector = this->primitive_->value_as_SparseToDense()->sparseValue();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> SparseToDense::GetDefaultValue() const {
  auto fb_vector = this->primitive_->value_as_SparseToDense()->defaultValue();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
bool SparseToDense::GetValidateIndices() const { return this->primitive_->value_as_SparseToDense()->validateIndices(); }
int SparseToDense::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_SparseToDense();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_SparseToDense return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> outputShape;
  if (attr->outputShape() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->outputShape()->size()); i++) {
      outputShape.push_back(attr->outputShape()->data()[i]);
    }
  }
  std::vector<int32_t> sparseValue;
  if (attr->sparseValue() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->sparseValue()->size()); i++) {
      sparseValue.push_back(attr->sparseValue()->data()[i]);
    }
  }
  std::vector<int32_t> defaultValue;
  if (attr->defaultValue() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->defaultValue()->size()); i++) {
      defaultValue.push_back(attr->defaultValue()->data()[i]);
    }
  }
  auto val_offset = schema::CreateSparseToDenseDirect(*fbb, &outputShape, &sparseValue, &defaultValue);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_SparseToDense, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif
}  // namespace lite
}  // namespace mindspore
