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

#include "src/ops/div.h"

#include "src/ops/ops_register.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Div::GetActivationType() const { return this->primitive_->value.AsDiv()->activationType; }

void Div::SetActivationType(int activation_type) {
  this->primitive_->value.AsDiv()->activationType = (schema::ActivationType)activation_type;
}

#else
int Div::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Div();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Div return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateDiv(*fbb, attr->activationType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Div, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int Div::GetActivationType() const { return this->primitive_->value_as_Div()->activationType(); }

PrimitiveC *DivCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Div>(primitive); }
Registry DivRegistry(schema::PrimitiveType_Div, DivCreator);
#endif
OpParameter *PopulateDivParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *arithmetic_param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(arithmetic_param, 0, sizeof(ArithmeticParameter));
  arithmetic_param->op_parameter_.type_ = primitive->Type();
  arithmetic_param->broadcasting_ = ((lite::Arithmetic *)primitive)->Broadcasting();
  arithmetic_param->ndim_ = ((lite::Arithmetic *)primitive)->NDims();
  arithmetic_param->activation_type_ =
    reinterpret_cast<mindspore::lite::Div *>(const_cast<mindspore::lite::PrimitiveC *>(primitive))->GetActivationType();
  auto tmp_shape = ((lite::Arithmetic *)primitive)->InShape0();
  memcpy(arithmetic_param->in_shape0_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  tmp_shape = ((lite::Arithmetic *)primitive)->InShape1();
  memcpy(arithmetic_param->in_shape1_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  tmp_shape = ((lite::Arithmetic *)primitive)->OutputShape();
  memcpy(arithmetic_param->out_shape_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}
Registry DivParameterRegistry(schema::PrimitiveType_Div, PopulateDivParameter);

}  // namespace lite
}  // namespace mindspore
