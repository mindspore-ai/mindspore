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

#include "src/ops/zeros_like.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {

#ifdef PRIMITIVE_WRITEABLE
#else
int ZerosLike::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);

  auto val_offset = schema::CreateZerosLike(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_ZerosLike, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *ZerosLikeCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<ZerosLike>(primitive);
}
Registry ZerosLikeRegistry(schema::PrimitiveType_ZerosLike, ZerosLikeCreator);

#endif

int ZerosLike::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (inputs_.size() != kSingleNum || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "zeroslike input or output number invalid, Input size:" << inputs_.size()
                  << ", output size: " << outputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  output->set_shape(input->shape());
  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore
