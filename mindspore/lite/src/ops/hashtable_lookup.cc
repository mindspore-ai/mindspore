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

#include "src/common/string_util.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

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
PrimitiveC *HashtableLookupCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<HashtableLookup>(primitive);
}
Registry HashtableLookupRegistry(schema::PrimitiveType_HashtableLookup, HashtableLookupCreator);
#endif

int HashtableLookup::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  auto input = inputs_.at(0);
  auto values = inputs_.at(2);
  auto output = outputs_.at(0);
  auto hits = outputs_.at(1);
  MS_ASSERT(input != nullptr);
  MS_ASSERT(values != nullptr);
  MS_ASSERT(output != nullptr);
  MS_ASSERT(hits != nullptr);

  std::vector<int> hits_shape;
  hits_shape.push_back(input->DimensionSize(0));

  output->set_data_type(values->data_type());
  output->set_format(input->format());
  hits->set_shape(hits_shape);
  hits->set_data_type(kNumberTypeUInt8);
  hits->set_format(input->format());

  if (input->data_c() == nullptr) {
    MS_LOG(INFO) << "Do infer shape in runtime.";
    return RET_INFER_INVALID;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
