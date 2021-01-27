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
#include "src/ops/tensorlist_setitem.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
TypeId TensorListSetItem::GetElementDType() const {
  return (TypeId)(this->primitive_->value.AsTensorListSetItem()->elementDType);
}

void TensorListSetItem::SetElementDType(int type) {
  this->primitive_->value.AsTensorListSetItem()->elementDType = type;
}

int TensorListSetItem::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_TensorListSetItem;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_TensorListSetItem) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::TensorListSetItemT();
    if (attr == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      MS_LOG(ERROR) << "new TensorListSetItemT value failed";
      return RET_ERROR;
    }
    if (prim.GetAttr("elementDType") == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      delete attr;
      MS_LOG(ERROR) << "TensorListSetItem's attr elementDType is not set";
      return RET_ERROR;
    } else {
      attr->elementDType = CastToInt(prim.GetAttr("elementDType")).front();
    }
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}
#else
TypeId TensorListSetItem::GetElementDType() const {
  return (TypeId)(this->primitive_->value_as_TensorListSetItem()->elementDType());
}

int TensorListSetItem::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_TensorListSetItem();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_TensorListSetItem return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateTensorListSetItem(*fbb, attr->elementDType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_TensorListSetItem, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *TensorListSetItemCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<TensorListSetItem>(primitive);
}
Registry TensorListSetItemRegistry(schema::PrimitiveType_TensorListSetItem, TensorListSetItemCreator);
#endif

int TensorListSetItem::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  auto input0 = reinterpret_cast<TensorList *>(inputs_[0]);
  MS_ASSERT(input0 != nullptr);
  auto get_index = inputs_[1];
  MS_ASSERT(get_index != nullptr);
  auto value_tensor = inputs_[2];
  MS_ASSERT(value_tensor != nullptr);
  auto output0 = reinterpret_cast<TensorList *>(outputs_[0]);
  MS_ASSERT(output0 != nullptr);

  output0->set_data_type(input0->data_type());
  output0->set_format(input0->format());

  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  if (get_index->data_c() == nullptr || value_tensor->data_c() == nullptr) {
    return RET_INFER_INVALID;
  }

  if (get_index->data_type() != kNumberTypeInt && get_index->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "inputs_[1]->data_type():" << get_index->data_type() << " is not int";
    return RET_ERROR;
  }
  if (get_index->ElementsNum() != 1) {
    MS_LOG(ERROR) << "inputs_[1].ElementsNum():" << get_index->ElementsNum() << " must be equal to 1!";
    return RET_ERROR;
  }
  if (get_index->data_c() == nullptr) {
    MS_LOG(ERROR) << "get_index->data_c() is nullptr";
    return RET_NULL_PTR;
  }
  int index = reinterpret_cast<int *>(get_index->data_c())[0];
  if (index < 0 || (index >= static_cast<int>(input0->tensors().size()) && index != 0)) {
    MS_LOG(ERROR) << "index_:" << index << "must in [0, " << input0->tensors().size() << "]";
    return RET_ERROR;
  }

  output0->set_max_elements_num(input0->max_elements_num());

  if (input0->tensors().empty() && input0->element_shape().empty() && index == 0) {
    input0->set_element_shape(value_tensor->shape());
    output0->set_element_shape(value_tensor->shape());
  } else {
    output0->set_element_shape(input0->element_shape());
  }
  std::vector<std::vector<int> > out_shape;
  if (index == 0 && input0->tensors().size() == 0) {  // uninitialized tensorlist
    out_shape.push_back(value_tensor->shape());
    output0->set_shape(std::vector<int>{1});
  } else {
    output0->set_shape(input0->shape());
    for (int i = 0; i < input0->ElementsNum(); ++i) {
      auto src_ptr = input0->GetTensor(i);
      if (src_ptr == nullptr) {
        MS_LOG(ERROR) << "input0->tensors_[" << i << "] is nullptr!";
        return RET_ERROR;
      }
      if (src_ptr->data_type() != kTypeUnknown) {
        out_shape.push_back(src_ptr->shape());
      } else {
        out_shape.push_back(std::vector<int>());
      }
    }
  }
  if (input0->tensors_data_type() == kTypeUnknown) {
    input0->set_tensors_data_type(value_tensor->data_type());
  }
  out_shape[index] = value_tensor->shape();
  output0->MallocTensorListData(input0->tensors_data_type(), out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
