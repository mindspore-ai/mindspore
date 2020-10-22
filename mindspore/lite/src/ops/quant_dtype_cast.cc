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

#include "src/ops/ops_register.h"
#include "nnacl/int8/quant_dtype_cast.h"

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

OpParameter *PopulateQuantDTypeCastParameter(const mindspore::lite::PrimitiveC *primitive) {
  QuantDTypeCastParameter *parameter =
    reinterpret_cast<QuantDTypeCastParameter *>(malloc(sizeof(QuantDTypeCastParameter)));
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "malloc QuantDTypeCastParameter failed.";
    return nullptr;
  }
  memset(parameter, 0, sizeof(QuantDTypeCastParameter));
  parameter->op_parameter_.type_ = primitive->Type();
  auto quant_dtype_cast_param =
    reinterpret_cast<mindspore::lite::QuantDTypeCast *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  parameter->srcT = quant_dtype_cast_param->GetSrcT();
  parameter->dstT = quant_dtype_cast_param->GetDstT();
  return reinterpret_cast<OpParameter *>(parameter);
}
Registry QuantDTypeCastParameterRegistry(schema::PrimitiveType_QuantDTypeCast, PopulateQuantDTypeCastParameter);

int QuantDTypeCast::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  MS_ASSERT(input->data_type() == param->srcT);
  output->set_data_type(static_cast<TypeId>(GetDstT()));
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  output->set_shape(input->shape());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
