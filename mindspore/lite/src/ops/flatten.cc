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

#include "src/ops/flatten.h"
#include <memory>

#include "src/ops/ops_register.h"
#include "nnacl/flatten.h"

namespace mindspore {
namespace lite {

int Flatten::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  auto output = outputs_.front();
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "Flatten input or output is null!";
    return RET_ERROR;
  }
  if (inputs_.size() != kSingleNum || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "input size: " << inputs_.size() << ", output size: " << outputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }

  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }

  auto input_shape = input->shape();
  std::vector<int> output_shape(2);
  output_shape[0] = input_shape[0];
  output_shape[1] = 1;
  for (size_t i = 1; i < input_shape.size(); i++) {
    output_shape[1] *= input_shape[i];
  }
  output->set_shape(output_shape);
  return RET_OK;
}
#ifdef PRIMITIVE_WRITEABLE
int Flatten::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Flatten;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Flatten) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::FlattenT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else
int Flatten::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateFlatten(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Flatten, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *FlattenCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Flatten>(primitive); }
Registry FlattenRegistry(schema::PrimitiveType_Flatten, FlattenCreator);
#endif

OpParameter *PopulateFlattenParameter(const mindspore::lite::PrimitiveC *primitive) {
  FlattenParameter *flatten_param = reinterpret_cast<FlattenParameter *>(malloc(sizeof(FlattenParameter)));
  if (flatten_param == nullptr) {
    MS_LOG(ERROR) << "malloc FlattenParameter failed.";
    return nullptr;
  }
  memset(flatten_param, 0, sizeof(FlattenParameter));
  flatten_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(flatten_param);
}

Registry FlattenParameterRegistry(schema::PrimitiveType_Flatten, PopulateFlattenParameter);

}  // namespace lite
}  // namespace mindspore
