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
#include "src/ops/tensorlist_reserve.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
TypeId TensorListReserve::GetElementDType() const {
  return (TypeId)(this->primitive_->value.AsTensorListReserve()->elementDType);
}

void TensorListReserve::SetElementDType(int type) {
  this->primitive_->value.AsTensorListReserve()->elementDType = type;
}

int TensorListReserve::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_TensorListReserve;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_TensorListReserve) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::TensorListReserveT();
    if (attr == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      MS_LOG(ERROR) << "new TensorListReserveT value failed";
      return RET_ERROR;
    }
    if (prim.GetAttr("elementDType") == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      delete attr;
      MS_LOG(ERROR) << "TensorListReserve's attr elementDType is not set";
      return RET_ERROR;
    } else {
      attr->elementDType = CastToInt(prim.GetAttr("elementDType")).front();
    }
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}

#else
TypeId TensorListReserve::GetElementDType() const {
  return (TypeId)(this->primitive_->value_as_TensorListReserve()->elementDType());
}

int TensorListReserve::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(primitive != nullptr);
  MS_ASSERT(fbb != nullptr);
  auto attr = primitive->value_as_TensorListReserve();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_TensorListReserve return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateTensorListReserve(*fbb, attr->elementDType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_TensorListReserve, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *TensorListReserveCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<TensorListReserve>(primitive);
}
Registry TensorListReserveRegistry(schema::PrimitiveType_TensorListReserve, TensorListReserveCreator);
#endif

int TensorListReserve::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  // input0: element_shape_tensor
  // input1: num_elements
  auto input0 = inputs_.front();
  MS_ASSERT(input0 != nullptr);
  auto ele_shape_type = input0->data_type();
  if (ele_shape_type != kNumberTypeInt && ele_shape_type != kNumberTypeInt32) {
    MS_LOG(ERROR) << "ele_shape_tensor.data_type():" << ele_shape_type << " is not int";
    return RET_ERROR;
  }
  if (input0->data_c() == nullptr) {
    MS_LOG(ERROR) << "input0->data_c() is nullptr";
    return RET_INFER_INVALID;
  }
  auto ele_shape_ptr = reinterpret_cast<int *>(input0->data_c());

  auto input1 = inputs_[1];
  MS_ASSERT(input1 != nullptr);
  auto num_ele_type = input1->data_type();
  if (num_ele_type != kNumberTypeInt && ele_shape_type != kNumberTypeInt32) {
    MS_LOG(ERROR) << "num_ele_tensor.data_type():" << num_ele_type << " is not int";
    return RET_ERROR;
  }
  if (input1->ElementsNum() != 1) {
    MS_LOG(ERROR) << "input1->ElementsNum() must be equal to 1";
    return RET_ERROR;
  }
  if (input1->data_c() == nullptr) {
    MS_LOG(ERROR) << "input1->data_c() is nullptr";
    return RET_INFER_INVALID;
  }
  int num_elements = reinterpret_cast<int *>(input1->data_c())[0];
  auto output = reinterpret_cast<TensorList *>(outputs_[0]);
  MS_ASSERT(output != nullptr);
  output->set_data_type(kObjectTypeTensorType);
  std::vector<std::vector<int> > tmp_shape(num_elements, std::vector<int>());
  output->set_element_shape(std::vector<int>(ele_shape_ptr, ele_shape_ptr + input0->ElementsNum()));
  output->set_shape(std::vector<int>(1, num_elements));
  output->MallocTensorListData(kTypeUnknown, tmp_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
