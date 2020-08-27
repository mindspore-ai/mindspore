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

#ifndef LITE_MINDSPORE_LITE_C_OPS_RESHAPE_H_
#define LITE_MINDSPORE_LITE_C_OPS_RESHAPE_H_

#include <vector>
#include <set>
#include <cmath>
#include <memory>
#include "ir/dtype/type_id.h"
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class Reshape : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(Reshape, PrimitiveC);
  Reshape() = default;
  explicit Reshape(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  int UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs);
  void SetFormat(int format);
  void SetShape(const std::vector<int64_t> &shape);
#else
  explicit Reshape(schema::Primitive *primitive) : PrimitiveC(primitive) {}

  schema::Primitive *Init(schema::Primitive *primitive) {
    flatbuffers::FlatBufferBuilder fbb(1024);

    auto attr = primitive->value_as_Reshape();
    MS_ASSERT(attr != nullptr);

    auto shape = std::make_unique<std::vector<int64_t>>();
    for (int i = 0; i < static_cast<int>(attr->shape()->size()); i++) {
      shape->push_back(attr->shape()->data()[i]);
    }

    auto val_offset = schema::CreateReshapeDirect(fbb, attr->format(), shape.release());
    auto prim_offset = schema::CreatePrimitive(fbb, schema::PrimitiveType_Reshape, val_offset.o);
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
  int GetFormat() const;
  std::vector<int64_t> GetShape() const;

 private:
  int CalNewShape(const lite::tensor::Tensor *in_tensor, std::vector<int> *out_shape) const;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_RESHAPE_H_
