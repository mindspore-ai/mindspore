/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/ops/hashtable_lookup.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int HashtableLookup::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) { return RET_OK; }
#else
int HashtableLookup::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateHashtableLookup(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_HashtableLookup, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif
int HashtableLookup::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  PrimitiveC::InferShape(inputs_, outputs_);
  return RET_INFER_INVALID;
}
}  // namespace lite
}  // namespace mindspore
