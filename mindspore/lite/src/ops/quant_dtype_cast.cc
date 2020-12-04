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

#include "src/ops/quant_dtype_cast.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int QuantDTypeCast::GetSrcT() const { return this->primitive_->value.AsQuantDTypeCast()->srcT; }
int QuantDTypeCast::GetDstT() const { return this->primitive_->value.AsQuantDTypeCast()->dstT; }

void QuantDTypeCast::SetSrcT(int src_t) { this->primitive_->value.AsQuantDTypeCast()->srcT = src_t; }
void QuantDTypeCast::SetDstT(int dst_t) { this->primitive_->value.AsQuantDTypeCast()->dstT = dst_t; }

#else

int QuantDTypeCast::GetSrcT() const { return this->primitive_->value_as_QuantDTypeCast()->srcT(); }
int QuantDTypeCast::GetDstT() const { return this->primitive_->value_as_QuantDTypeCast()->dstT(); }
int QuantDTypeCast::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_QuantDTypeCast();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_QuantDTypeCast return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateQuantDTypeCast(*fbb, attr->srcT(), attr->dstT());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_QuantDTypeCast, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *QuantDTypeCastCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<QuantDTypeCast>(primitive);
}
Registry QuantDTypeCastRegistry(schema::PrimitiveType_QuantDTypeCast, QuantDTypeCastCreator);
#endif

int QuantDTypeCast::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  MS_ASSERT(input->data_type() == this->GetSrcT());
  output->set_data_type(static_cast<TypeId>(GetDstT()));
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  output->set_shape(input->shape());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
