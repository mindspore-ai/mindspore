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

#ifndef LITE_MINDSPORE_LITE_C_OPS_BROADCAST_TO_H_
#define LITE_MINDSPORE_LITE_C_OPS_BROADCAST_TO_H_

#include <vector>
#include <set>
#include <cmath>
#include <memory>
#include "ir/dtype/type_id.h"
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class BroadcastTo : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(BroadcastTo, PrimitiveC);
  BroadcastTo() = default;
  explicit BroadcastTo(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  void SetDstShape(const std::vector<int> &dst_shape);

#else
  explicit BroadcastTo(schema::Primitive *primitive) : PrimitiveC(primitive) {}

  schema::Primitive *Init(schema::Primitive *primitive) {
    flatbuffers::FlatBufferBuilder fbb(1024);

    auto attr = primitive->value_as_BroadcastTo();
    MS_ASSERT(attr != nullptr);

    auto dst_shape = std::make_unique<std::vector<int32_t>>();
    for (int i = 0; i < static_cast<int>(attr->dst_shape()->size()); i++) {
      dst_shape->push_back(attr->dst_shape()->data()[i]);
    }

    auto val_offset = schema::CreateBroadcastToDirect(fbb, dst_shape.release());
    auto prim_offset = schema::CreatePrimitive(fbb, schema::PrimitiveType_BroadcastTo, val_offset.o);
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
  std::vector<int> GetDstShape() const;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_BROADCAST_TO_H_
