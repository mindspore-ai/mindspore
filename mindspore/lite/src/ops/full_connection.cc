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

#include "src/ops/full_connection.h"

#include "src/ops/ops_register.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
bool FullConnection::GetHasBias() const { return this->primitive_->value.AsFullConnection()->hasBias; }
int FullConnection::GetAxis() const { return this->primitive_->value.AsFullConnection()->axis; }
bool FullConnection::GetUseAxis() const { return this->primitive_->value.AsFullConnection()->useAxis; }
int FullConnection::GetActivationType() const { return this->primitive_->value.AsFullConnection()->activationType; }

void FullConnection::SetHasBias(bool has_bias) { this->primitive_->value.AsFullConnection()->hasBias = has_bias; }
void FullConnection::SetAxis(int axis) { this->primitive_->value.AsFullConnection()->axis = axis; }
void FullConnection::SetUseAxis(bool use_axis) { this->primitive_->value.AsFullConnection()->useAxis = use_axis; }
void FullConnection::SetActivationType(int activationType) {
  this->primitive_->value.AsFullConnection()->activationType = (schema::ActivationType)activationType;
}
#else
int FullConnection::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_FullConnection();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_FullConnection return nullptr";
    return RET_ERROR;
  }

  auto val_offset =
    schema::CreateFullConnection(*fbb, attr->hasBias(), attr->axis(), attr->useAxis(), attr->activationType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_FullConnection, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
bool FullConnection::GetHasBias() const { return this->primitive_->value_as_FullConnection()->hasBias(); }
int FullConnection::GetAxis() const { return this->primitive_->value_as_FullConnection()->axis(); }
bool FullConnection::GetUseAxis() const { return this->primitive_->value_as_FullConnection()->useAxis(); }
int FullConnection::GetActivationType() const { return this->primitive_->value_as_FullConnection()->activationType(); }

PrimitiveC *FullConnectionCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<FullConnection>(primitive);
}
Registry FullConnectionRegistry(schema::PrimitiveType_FullConnection, FullConnectionCreator);
#endif

OpParameter *PopulateFullconnectionParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto param =
    reinterpret_cast<mindspore::lite::FullConnection *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  MatMulParameter *matmul_param = reinterpret_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "malloc MatMulParameter failed.";
    return nullptr;
  }
  memset(matmul_param, 0, sizeof(MatMulParameter));
  matmul_param->op_parameter_.type_ = primitive->Type();
  matmul_param->b_transpose_ = true;
  matmul_param->a_transpose_ = false;
  matmul_param->has_bias_ = param->GetHasBias();
  if (param->GetActivationType() == schema::ActivationType_RELU) {
    matmul_param->act_type_ = ActType_Relu;
  } else if (param->GetActivationType() == schema::ActivationType_RELU6) {
    matmul_param->act_type_ = ActType_Relu6;
  } else {
    matmul_param->act_type_ = ActType_No;
  }

  return reinterpret_cast<OpParameter *>(matmul_param);
}

Registry FullConnectionParameterRegistry(schema::PrimitiveType_FullConnection, PopulateFullconnectionParameter);

int FullConnection::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input0 = inputs_.front();
  MS_ASSERT(input0 != nullptr);
  auto input1 = inputs_[1];
  MS_ASSERT(input1 != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (!GetInferFlag()) {
    return RET_OK;
  }
  if ((GetHasBias() && inputs_.size() != kMultiNum) || (!GetHasBias() && inputs_.size() != kDoubleNum)) {
    MS_LOG(ERROR) << "Input tensors num error";
    return RET_INPUT_TENSOR_ERROR;
  }
  if (GetUseAxis() && (GetAxis() < 1 || GetAxis() > static_cast<int>(input0->shape().size()))) {
    MS_LOG(ERROR) << "FullConnection axis invalid";
    return RET_ERROR;
  }
  int new_k = 1;
  if (GetUseAxis()) {
    for (size_t i = GetAxis(); i < input0->shape().size(); ++i) {
      new_k *= input0->shape()[i];
    }
    if (new_k != input1->shape()[1]) {
      MS_LOG(ERROR) << "Input1 size invalid";
      return RET_INPUT_TENSOR_ERROR;
    }
  } else {
    new_k = input1->shape()[1];
  }
  if (GetHasBias()) {
    if (inputs_[2]->shape()[0] != input1->shape()[0]) {
      MS_LOG(ERROR) << "bias size invalid";
      return RET_INPUT_TENSOR_ERROR;
    }
  }
  std::vector<int> out_shape{inputs_[0]->shape()};
  if (GetUseAxis()) {
    out_shape.resize(GetAxis() + 1);
    out_shape[GetAxis()] = input1->shape()[0];
  } else {
    int total = 1;
    for (size_t i = 0; i < input0->shape().size(); ++i) {
      total *= input0->shape()[i];
    }
    out_shape.resize(2);
    auto batch_size = total / new_k;
    out_shape[0] = batch_size;
    out_shape[1] = input1->shape()[0];
  }
  output->set_shape(out_shape);
  output->set_data_type(input0->data_type());
  output->SetFormat(input0->GetFormat());

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
