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

#ifndef LITE_MINDSPORE_LITE_SRC_OPS_CONSTANT_OF_SHAPE_H_
#define LITE_MINDSPORE_LITE_SRC_OPS_CONSTANT_OF_SHAPE_H_

#include <vector>
#include <set>
#include <cmath>
#include "ir/dtype/type_id.h"
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class ConstantOfShape : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(ConstantOfShape, PrimitiveC);
  ConstantOfShape() = default;
  explicit ConstantOfShape(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  void SetValue(float value);
#else
  explicit ConstantOfShape(schema::Primitive *primitive) : PrimitiveC(primitive) {}

  schema::Primitive *Init(schema::Primitive *primitive) {
    flatbuffers::FlatBufferBuilder fbb(1024);

    auto attr = primitive->value_as_ConstantOfShape();
    MS_ASSERT(attr != nullptr);

    auto val_offset = schema::CreateConstantOfShape(fbb, attr->value());
    auto prim_offset = schema::CreatePrimitive(fbb, schema::PrimitiveType_ConstantOfShape, val_offset.o);
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
  int InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) override;
  float GetValue() const;
};
}  // namespace lite
}  // namespace mindspore
#endif  // LITE_MINDSPORE_LITE_SRC_OPS_CONSTANT_OF_SHAPE_H_
