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

#include "c_ops/sparse_to_dense.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> SparseToDense::GetOutputShape() const { return this->primitive->value.AsSparseToDense()->outputShape; }
std::vector<int> SparseToDense::GetSparseValue() const { return this->primitive->value.AsSparseToDense()->sparseValue; }
std::vector<int> SparseToDense::GetDefaultValue() const {
  return this->primitive->value.AsSparseToDense()->defaultValue;
}
bool SparseToDense::GetValidateIndices() const { return this->primitive->value.AsSparseToDense()->validateIndices; }

void SparseToDense::SetOutputShape(const std::vector<int> &output_shape) {
  this->primitive->value.AsSparseToDense()->outputShape = output_shape;
}
void SparseToDense::SetSparseValue(const std::vector<int> &sparse_value) {
  this->primitive->value.AsSparseToDense()->sparseValue = sparse_value;
}
void SparseToDense::SetDefaultValue(const std::vector<int> &default_value) {
  this->primitive->value.AsSparseToDense()->defaultValue = default_value;
}
void SparseToDense::SetValidateIndices(bool validate_indices) {
  this->primitive->value.AsSparseToDense()->validateIndices = validate_indices;
}

#else

std::vector<int> SparseToDense::GetOutputShape() const {
  auto fb_vector = this->primitive->value_as_SparseToDense()->outputShape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> SparseToDense::GetSparseValue() const {
  auto fb_vector = this->primitive->value_as_SparseToDense()->sparseValue();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> SparseToDense::GetDefaultValue() const {
  auto fb_vector = this->primitive->value_as_SparseToDense()->defaultValue();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
bool SparseToDense::GetValidateIndices() const { return this->primitive->value_as_SparseToDense()->validateIndices(); }

void SparseToDense::SetOutputShape(const std::vector<int> &output_shape) {}
void SparseToDense::SetSparseValue(const std::vector<int> &sparse_value) {}
void SparseToDense::SetDefaultValue(const std::vector<int> &default_value) {}
void SparseToDense::SetValidateIndices(bool validate_indices) {}
#endif
}  // namespace mindspore
