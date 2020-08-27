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

#ifndef LITE_MINDSPORE_LITE_C_OPS_SPARSE_TO_DENSE_H_
#define LITE_MINDSPORE_LITE_C_OPS_SPARSE_TO_DENSE_H_

#include <vector>
#include <set>
#include <cmath>
#include <memory>
#include "ir/dtype/type_id.h"
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class SparseToDense : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(SparseToDense, PrimitiveC);
  SparseToDense() = default;
  explicit SparseToDense(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  void SetOutputShape(const std::vector<int> &output_shape);
  void SetSparseValue(const std::vector<int> &sparse_value);
  void SetDefaultValue(const std::vector<int> &default_value);
  void SetValidateIndices(bool validate_indices);
#else
  explicit SparseToDense(schema::Primitive *primitive) : PrimitiveC(primitive) {}

  schema::Primitive *Init(schema::Primitive *primitive) {
    flatbuffers::FlatBufferBuilder fbb(1024);

    auto attr = primitive->value_as_SparseToDense();
    MS_ASSERT(attr != nullptr);

    auto outputShape = std::make_unique<std::vector<int32_t>>();
    for (int i = 0; i < static_cast<int>(attr->outputShape()->size()); i++) {
      outputShape->push_back(attr->outputShape()->data()[i]);
    }
    auto sparseValue = std::make_unique<std::vector<int32_t>>();
    for (int i = 0; i < static_cast<int>(attr->sparseValue()->size()); i++) {
      sparseValue->push_back(attr->sparseValue()->data()[i]);
    }
    auto defaultValue = std::make_unique<std::vector<int32_t>>();
    for (int i = 0; i < static_cast<int>(attr->defaultValue()->size()); i++) {
      defaultValue->push_back(attr->defaultValue()->data()[i]);
    }

    auto val_offset = schema::CreateSparseToDenseDirect(fbb, outputShape.release(),
                                                  sparseValue.release(), defaultValue.release());
    auto prim_offset = schema::CreatePrimitive(fbb, schema::PrimitiveType_SparseToDense, val_offset.o);
    fbb.Finish(prim_offset);

    auto buf = fbb.GetBufferPointer();
    MS_ASSERT(buf != nullptr);
    auto buf_bak = new char[fbb.GetSize()];
    memcpy(buf_bak, buf, fbb.GetSize());

    auto root = flatbuffers::GetRoot<schema::Primitive>(buf_bak);
    auto prim = const_cast<schema::Primitive *>(root);

    delete[] buf_bak;
    fbb.Clear();
    return prim;
  }
#endif
  std::vector<int> GetOutputShape() const;
  std::vector<int> GetSparseValue() const;
  std::vector<int> GetDefaultValue() const;
  bool GetValidateIndices() const;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_SPARSE_TO_DENSE_H_
