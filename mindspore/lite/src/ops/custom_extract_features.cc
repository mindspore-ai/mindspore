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
#include "src/ops/custom_extract_features.h"

#include "src/common/string_util.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int CustomExtractFeatures::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) { return RET_OK; }
#else
int CustomExtractFeatures::UnPackToFlatBuilder(const schema::Primitive *primitive,
                                               flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateCustomExtractFeatures(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_CustomExtractFeatures, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *CustomExtractFeaturesCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<CustomExtractFeatures>(primitive);
}
Registry CustomExtractFeaturesRegistry(schema::PrimitiveType_CustomExtractFeatures, CustomExtractFeaturesCreator);
#endif

int CustomExtractFeatures::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  auto input = inputs_.at(0);
  auto output0 = outputs_.at(0);
  auto output1 = outputs_.at(1);
  MS_ASSERT(input != nullptr);
  MS_ASSERT(output0 != nullptr);
  MS_ASSERT(output1 != nullptr);

  output0->set_data_type(kNumberTypeInt32);
  output0->set_format(input->format());
  output1->set_data_type(kNumberTypeFloat32);
  output1->set_format(input->format());

  if (input->data_c() == nullptr) {
    MS_LOG(INFO) << "Do infer shape in runtime.";
    return RET_INFER_INVALID;
  }
  std::vector<int> shape;
  int string_num = lite::GetStringCount(input);
  shape.push_back(string_num == 0 ? 1 : string_num);

  output0->set_shape(shape);
  output1->set_shape(shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
