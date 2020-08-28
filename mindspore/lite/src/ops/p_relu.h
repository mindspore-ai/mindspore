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

#ifndef LITE_MINDSPORE_LITE_C_OPS_P_RELU_H_
#define LITE_MINDSPORE_LITE_C_OPS_P_RELU_H_

#include <vector>
#include <set>
#include <cmath>
#include <memory>
#include "ir/dtype/type_id.h"
#include "src/ops/activation.h"

namespace mindspore {
namespace lite {
class PReLU : public Activation {
 public:
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(PReLU, Activation);
  PReLU() = default;
  explicit PReLU(schema::PrimitiveT *primitive) : Activation(primitive) {}
  void SetChannelShared(bool channel_shared);

#else
  explicit PReLU(schema::Primitive *primitive) : Activation(primitive) {}

  schema::Primitive *Init(schema::Primitive *primitive) {
    flatbuffers::FlatBufferBuilder fbb(1024);

    auto attr = primitive->value_as_PReLU();
    MS_ASSERT(attr != nullptr);

    auto slope = std::make_unique<std::vector<float>>();
    for (int i = 0; i < static_cast<int>(attr->slope()->size()); i++) {
      slope->push_back(attr->slope()->data()[i]);
    }

    auto val_offset = schema::CreatePReLUDirect(fbb, attr->channelShared(), slope.release());
    auto prim_offset = schema::CreatePrimitive(fbb, schema::PrimitiveType_PReLU, val_offset.o);
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
  bool GetChannelShared() const;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_P_RELU_H_
