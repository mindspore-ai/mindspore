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

#include "src/ops/p_relu.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
bool PReLU::GetChannelShared() const { return this->primitive_->value.AsPReLU()->channelShared; }

void PReLU::SetChannelShared(bool channel_shared) { this->primitive_->value.AsPReLU()->channelShared = channel_shared; }

#else
bool PReLU::GetChannelShared() const { return this->primitive_->value_as_PReLU()->channelShared(); }

int PReLU::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_PReLU();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_PReLU return nullptr";
    return RET_ERROR;
  }
  std::vector<float> slope;
  if (attr->slope() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->slope()->size()); i++) {
      slope.push_back(attr->slope()->data()[i]);
    }
  }
  auto val_offset = schema::CreatePReLUDirect(*fbb, attr->channelShared(), &slope);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_PReLU, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

#endif
}  // namespace lite
}  // namespace mindspore
