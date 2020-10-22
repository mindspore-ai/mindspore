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

#include "src/ops/add.h"
#include <memory>
#include "src/ops/ops_register.h"
#include "nnacl/arithmetic_common.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Add::GetActivationType() const { return this->primitive_->value.AsAdd()->activationType; }

void Add::SetActivationType(int activation_type) {
  this->primitive_->value.AsAdd()->activationType = (schema::ActivationType)activation_type;
}

int Add::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Add;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Add) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    this->primitive_->value.value = new (std::nothrow) schema::AddT();
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  if (GetQuantType() == schema::QuantType_AwareTraining) {
    std::vector<std::vector<schema::QuantParamT>> vecInputQuantParam;
    std::vector<std::vector<schema::QuantParamT>> vecOutputQuantParam;
    PopulaterQuantParam(prim, &vecInputQuantParam, &vecOutputQuantParam, inputs);
    SetOutputQuantParam(vecOutputQuantParam);
  }
  return RET_OK;
}

#else
int Add::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Add();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Add return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateAdd(*fbb, attr->activationType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Add, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int Add::GetActivationType() const { return this->primitive_->value_as_Add()->activationType(); }

PrimitiveC *AddCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Add>(primitive); }
Registry AddRegistry(schema::PrimitiveType_Add, AddCreator);
#endif

OpParameter *PopulateAddParameter(const mindspore::lite::PrimitiveC *primitive) {
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
    reinterpret_cast<mindspore::lite::Add *>(const_cast<mindspore::lite::PrimitiveC *>(primitive))->GetActivationType();
  auto tmp_shape = ((lite::Arithmetic *)primitive)->InShape0();
  memcpy(arithmetic_param->in_shape0_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  tmp_shape = ((lite::Arithmetic *)primitive)->InShape1();
  memcpy(arithmetic_param->in_shape1_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  tmp_shape = ((lite::Arithmetic *)primitive)->OutputShape();
  memcpy(arithmetic_param->out_shape_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}
Registry AddParameterRegistry(schema::PrimitiveType_Add, PopulateAddParameter);

}  // namespace lite
}  // namespace mindspore
