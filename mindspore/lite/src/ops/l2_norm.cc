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

#include "src/ops/l2_norm.h"

#include "src/ops/ops_register.h"
#include "nnacl/l2_norm_parameter.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> L2Norm::GetAxis() const { return this->primitive_->value.AsL2Norm()->axis; }
float L2Norm::GetEpsilon() const { return this->primitive_->value.AsL2Norm()->epsilon; }
int L2Norm::GetActivationType() const { return this->primitive_->value.AsL2Norm()->activationType; }

void L2Norm::SetAxis(const std::vector<int> &axis) { this->primitive_->value.AsL2Norm()->axis = axis; }
void L2Norm::SetEpsilon(float epsilon) { this->primitive_->value.AsL2Norm()->epsilon = epsilon; }
void L2Norm::SetActivationType(int activationType) {
  this->primitive_->value.AsL2Norm()->activationType = (schema::ActivationType)activationType;
}

#else
int L2Norm::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_L2Norm();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_L2Norm return nullptr";
    return RET_ERROR;
  }

  std::vector<int32_t> axis;
  if (attr->axis() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->axis()->size()); i++) {
      axis.push_back(attr->axis()->data()[i]);
    }
  }
  auto val_offset = schema::CreateL2NormDirect(*fbb, &axis, attr->epsilon());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_L2Norm, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
std::vector<int> L2Norm::GetAxis() const {
  auto fb_vector = this->primitive_->value_as_L2Norm()->axis();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
float L2Norm::GetEpsilon() const { return this->primitive_->value_as_L2Norm()->epsilon(); }
int L2Norm::GetActivationType() const { return this->primitive_->value_as_L2Norm()->activationType(); }

PrimitiveC *L2NormCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<L2Norm>(primitive); }
Registry L2NormRegistry(schema::PrimitiveType_L2Norm, L2NormCreator);
#endif
OpParameter *PopulateL2NormParameter(const mindspore::lite::PrimitiveC *primitive) {
  L2NormParameter *l2_norm_parameter = reinterpret_cast<L2NormParameter *>(malloc(sizeof(L2NormParameter)));
  if (l2_norm_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc L2NormParameter failed.";
    return nullptr;
  }
  memset(l2_norm_parameter, 0, sizeof(L2NormParameter));
  l2_norm_parameter->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::L2Norm *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto axis_vec = param->GetAxis();
  l2_norm_parameter->axis_num_ = axis_vec.size();
  l2_norm_parameter->axis_ = reinterpret_cast<int *>(malloc(axis_vec.size() * sizeof(int)));
  if (l2_norm_parameter->axis_ == nullptr) {
    MS_LOG(ERROR) << "malloc axis_ data failed";
    free(l2_norm_parameter);
    return nullptr;
  }
  for (size_t i = 0; i < axis_vec.size(); i++) {
    l2_norm_parameter->axis_[i] = axis_vec[i];
  }
  if (param->GetEpsilon() < 1e-6) {
    l2_norm_parameter->epsilon_ = 1e-6;
  } else {
    l2_norm_parameter->epsilon_ = param->GetEpsilon();
  }
  if (param->GetActivationType() == static_cast<int>(schema::ActivationType_RELU)) {
    l2_norm_parameter->act_type_ = ActType_Relu;
  } else if (param->GetActivationType() == static_cast<int>(schema::ActivationType_RELU6)) {
    l2_norm_parameter->act_type_ = ActType_Relu6;
  } else {
    l2_norm_parameter->act_type_ = ActType_No;
  }
  return reinterpret_cast<OpParameter *>(l2_norm_parameter);
}
Registry L2NormParameterRegistry(schema::PrimitiveType_L2Norm, PopulateL2NormParameter);

}  // namespace lite
}  // namespace mindspore
