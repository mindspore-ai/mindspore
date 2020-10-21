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

#include "src/ops/eltwise.h"

#include "src/ops/ops_register.h"
#include "nnacl/arithmetic_common.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Eltwise::GetMode() const { return this->primitive_->value.AsEltwise()->mode; }

void Eltwise::SetMode(int mode) { this->primitive_->value.AsEltwise()->mode = (schema::EltwiseMode)mode; }

#else
int Eltwise::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Eltwise();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Eltwise return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateEltwise(*fbb, attr->mode());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Eltwise, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int Eltwise::GetMode() const { return this->primitive_->value_as_Eltwise()->mode(); }

PrimitiveC *EltwiseCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Eltwise>(primitive); }
Registry EltwiseRegistry(schema::PrimitiveType_Eltwise, EltwiseCreator);
#endif

OpParameter *PopulateEltwiseParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *arithmetic_param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(arithmetic_param, 0, sizeof(ArithmeticParameter));
  auto eltwise = reinterpret_cast<mindspore::lite::Eltwise *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  switch (eltwise->GetMode()) {
    case schema::EltwiseMode_PROD:
      arithmetic_param->op_parameter_.type_ = schema::PrimitiveType_Mul;
      break;
    case schema::EltwiseMode_SUM:
      arithmetic_param->op_parameter_.type_ = schema::PrimitiveType_Add;
      break;
    case schema::EltwiseMode_MAXIMUM:
      arithmetic_param->op_parameter_.type_ = schema::PrimitiveType_Maximum;
      break;
    default:
      free(arithmetic_param);
      return nullptr;
  }
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}
Registry EltwiseParameterRegistry(schema::PrimitiveType_Eltwise, PopulateEltwiseParameter);

}  // namespace lite
}  // namespace mindspore
