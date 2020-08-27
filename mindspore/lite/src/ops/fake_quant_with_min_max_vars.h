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

#ifndef LITE_MINDSPORE_LITE_C_OPS_FAKE_QUANT_WITH_MIN_MAX_VARS_H_
#define LITE_MINDSPORE_LITE_C_OPS_FAKE_QUANT_WITH_MIN_MAX_VARS_H_

#include <vector>
#include <set>
#include <cmath>
#include "ir/dtype/type_id.h"
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class FakeQuantWithMinMaxVars : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(FakeQuantWithMinMaxVars, PrimitiveC);
  FakeQuantWithMinMaxVars() = default;
  explicit FakeQuantWithMinMaxVars(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  void SetNarrowRange(bool narrow_range);
  void SetNumBits(int num_bits);
#else
  explicit FakeQuantWithMinMaxVars(schema::Primitive *primitive) : PrimitiveC(primitive) {}

  schema::Primitive *Init(schema::Primitive *primitive) {
    flatbuffers::FlatBufferBuilder fbb(1024);

    auto attr = primitive->value_as_FakeQuantWithMinMaxVars();
    MS_ASSERT(attr != nullptr);

    auto val_offset = schema::CreateFakeQuantWithMinMaxVars(fbb, attr->narrowRange(), attr->numBits());
    auto prim_offset = schema::CreatePrimitive(fbb,
                                               schema::PrimitiveType_FakeQuantWithMinMaxVars, val_offset.o);
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
  bool GetNarrowRange() const;
  int GetNumBits() const;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_FAKE_QUANT_WITH_MIN_MAX_VARS_H_
