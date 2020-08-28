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

#include "src/ops/upsample.h"
#include <string>

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::string Upsample::GetMode() const { return this->primitive_->value.AsUpsample()->mode; }
std::vector<float> Upsample::GetScales() const { return this->primitive_->value.AsUpsample()->scales; }

void Upsample::SetMode(std::string mode) { this->primitive_->value.AsUpsample()->mode = mode; }
void Upsample::SetScales(const std::vector<float> &scales) { this->primitive_->value.AsUpsample()->scales = scales; }

#else

std::string Upsample::GetMode() const { return this->primitive_->value_as_Upsample()->mode()->str(); }
std::vector<float> Upsample::GetScales() const {
  auto fb_vector = this->primitive_->value_as_Upsample()->scales();
  return std::vector<float>(fb_vector->begin(), fb_vector->end());
}
int Upsample::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Upsample();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Upsample return nullptr";
    return RET_ERROR;
  }
  std::vector<float> scales;
  if (attr->scales() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->scales()->size()); i++) {
      scales.push_back(attr->scales()->data()[i]);
    }
  }
  auto val_offset = schema::CreateUpsampleDirect(*fbb, attr->mode()->c_str(), &scales);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Upsample, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif
}  // namespace lite
}  // namespace mindspore
