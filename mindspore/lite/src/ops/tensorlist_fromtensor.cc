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
#include <vector>
#include "src/ops/tensorlist_fromtensor.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int TensorListFromTensor::GetElementDType() const {
  return this->primitive_->value.AsTensorListFromTensor()->elementDType;
}

int TensorListFromTensor::GetShapeType() const { return this->primitive_->value.AsTensorListFromTensor()->shapeType; }

void TensorListFromTensor::SetElementDType(int type) {
  this->primitive_->value.AsTensorListFromTensor()->elementDType = type;
}

void TensorListFromTensor::SetShapeType(int type) {
  this->primitive_->value.AsTensorListFromTensor()->shapeType = type;
}

int TensorListFromTensor::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_TensorListFromTensor;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_TensorListFromTensor) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::TensorListFromTensorT();
    if (attr == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      MS_LOG(ERROR) << "new TensorListFromTensorT value failed";
      return RET_ERROR;
    }
    if (prim.GetAttr("elementDType") == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      delete attr;
      MS_LOG(ERROR) << "TensorListFromTensorT's attr elementDType is not set";
      return RET_ERROR;
    } else {
      attr->elementDType = CastToInt(prim.GetAttr("elementDType")).front();
    }
    if (prim.GetAttr("shapeType") == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      delete attr;
      MS_LOG(ERROR) << "TensorListFromTensorT's attr shapeType is not set";
      return RET_ERROR;
    } else {
      attr->shapeType = CastToInt(prim.GetAttr("shapeType")).front();
    }
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}
#else
int TensorListFromTensor::GetElementDType() const {
  return this->primitive_->value_as_TensorListFromTensor()->elementDType();
}

int TensorListFromTensor::GetShapeType() const {
  return this->primitive_->value_as_TensorListFromTensor()->shapeType();
}

int TensorListFromTensor::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_TensorListFromTensor();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_TensorListFromTensor return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateTensorListFromTensor(*fbb, attr->elementDType(), attr->shapeType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_TensorListFromTensor, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *TensorListFromTensorCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<TensorListFromTensor>(primitive);
}
Registry TensorListFromTensorRegistry(schema::PrimitiveType_TensorListFromTensor, TensorListFromTensorCreator);
#endif

int TensorListFromTensor::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto input0 = inputs_[0];
  MS_ASSERT(input0 != nullptr);
  std::vector<int> input0_shape = input0->shape();
  if (input0_shape.size() < 1) {
    MS_LOG(ERROR) << "input0_shape.size():" << input0_shape.size() << " must be greater than 0!";
    return RET_ERROR;
  }
  int dim0 = input0_shape[0];
  if (dim0 < 0) {
    MS_LOG(ERROR) << "inputs_[0] dim0:" << dim0 << " must greater than or equal to 0";
    return RET_ERROR;
  }
  auto input1 = inputs_[1];
  MS_ASSERT(input1 != nullptr);
  if (input1->data_c() == nullptr) {
    MS_LOG(ERROR) << "input1->data_c() is nullptr";
    return RET_NULL_PTR;
  }
  auto ele_shape_ptr = reinterpret_cast<int *>(input1->data_c());
  auto output = reinterpret_cast<TensorList *>(outputs_[0]);
  MS_ASSERT(output != nullptr);
  std::vector<std::vector<int> > tensor_shape(dim0, std::vector<int>(input0_shape.begin() + 1, input0_shape.end()));
  output->set_element_shape(std::vector<int>(ele_shape_ptr, ele_shape_ptr + input1->ElementsNum()));
  output->set_shape(std::vector<int>(1, dim0));
  output->set_data_type(kObjectTypeTensorType);
  output->MallocTensorListData(input0->data_type(), tensor_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
