/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/ops/invert_permutation.h"
#include "src/common/common.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {

#ifdef PRIMITIVE_WRITEABLE
#else
int InvertPermutation::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateSize(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_InvertPermutation, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *InvertPermutationCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<InvertPermutation>(primitive);
}
Registry InvertPermutationRegistry(schema::PrimitiveType_InvertPermutation, InvertPermutationCreator);
#endif

int InvertPermutation::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_format(input->format());
  output->set_data_type(input->data_type());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  if (input->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "InvertPermutation does not support input of data type: " << input->data_type();
    return RET_ERROR;
  }
  if (input->shape().size() != 1) {
    MS_LOG(ERROR) << "InvertPermutation input must be one-dimensional.";
    return RET_ERROR;
  }
  output->set_shape(input->shape());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
