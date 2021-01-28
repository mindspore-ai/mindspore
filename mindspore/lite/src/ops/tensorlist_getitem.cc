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
#include "src/ops/tensorlist_getitem.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
TypeId TensorListGetItem::GetElementDType() const {
  return (TypeId)(this->primitive_->value.AsTensorListGetItem()->elementDType);
}

void TensorListGetItem::SetElementDType(int type) {
  this->primitive_->value.AsTensorListGetItem()->elementDType = type;
}

int TensorListGetItem::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_TensorListGetItem;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_TensorListGetItem) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::TensorListGetItemT();
    if (attr == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      MS_LOG(ERROR) << "new TensorListGetItemT value failed";
      return RET_ERROR;
    }
    if (prim.GetAttr("elementDType") == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      delete attr;
      MS_LOG(ERROR) << "TensorListGetItem's attr elementDType is not set";
      return RET_ERROR;
    } else {
      attr->elementDType = CastToInt(prim.GetAttr("elementDType")).front();
    }
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}
#else
TypeId TensorListGetItem::GetElementDType() const {
  return (TypeId)(this->primitive_->value_as_TensorListGetItem()->elementDType());
}

int TensorListGetItem::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_TensorListGetItem();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_TensorListGetItem return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateTensorListGetItem(*fbb, attr->elementDType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_TensorListGetItem, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *TensorListGetItemCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<TensorListGetItem>(primitive);
}
Registry TensorListGetItemRegistry(schema::PrimitiveType_TensorListGetItem, TensorListGetItemCreator);
#endif
bool TensorListGetItem::IsFullyDefined(const std::vector<int> &shape) const {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      return false;
    }
  }
  return true;
}

int TensorListGetItem::MergeShape(const std::vector<int> &tmp) {
  if (element_shape_.size() != tmp.size()) {
    MS_LOG(ERROR) << "element_shape_.size():" << element_shape_.size() << " must be equal to tmp.size():" << tmp.size();
    return RET_ERROR;
  }
  for (size_t j = 0; j < tmp.size(); ++j) {
    if (element_shape_[j] >= 0 && tmp[j] >= 0 && element_shape_[j] != tmp[j]) {
      MS_LOG(ERROR) << "element_shape_[" << j << "]:" << element_shape_[j] << " must be equal to tmp[" << j
                    << "]:" << tmp[j];
      return RET_ERROR;
    }
    element_shape_[j] = element_shape_[j] >= 0 ? element_shape_[j] : tmp[j];
  }
  return RET_OK;
}

int TensorListGetItem::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  MS_ASSERT(inputs_.size() >= 3);
  MS_ASSERT(inputs_.at(0) != nullptr);
  MS_ASSERT(inputs_.at(1) != nullptr);
  MS_ASSERT(inputs_.at(2) != nullptr);
  auto input0 = reinterpret_cast<TensorList *>(inputs_.at(0));
  auto get_index = inputs_.at(1);
  MS_ASSERT(get_index != nullptr);
  if (get_index->ElementsNum() != 1) {
    MS_LOG(ERROR) << "get_index->ElementsNum():" << get_index->ElementsNum() << " must be equal to 1!";
    return RET_ERROR;
  }
  if (get_index->data_c() == nullptr) {
    MS_LOG(DEBUG) << "get_index->data_c() is nullptr";
    return RET_INFER_INVALID;
  }
  index_ = reinterpret_cast<int *>(get_index->data_c())[0];
  if (index_ < 0 || index_ > (input0->ElementsNum() - 1)) {
    MS_LOG(ERROR) << "index_:" << index_ << "must in [0, " << input0->ElementsNum() - 1 << "]";
    return RET_ERROR;
  }
  auto tensor_index = input0->GetTensor(index_);
  if (tensor_index == nullptr) {
    return RET_INFER_INVALID;
  }
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (tensor_index->data_type() != kTypeUnknown) {
    output->set_data_type(tensor_index->data_type());
    output->set_shape(tensor_index->shape());
  } else {
    auto input2 = inputs_[2];
    if (input2->data_c() == nullptr) {
      MS_LOG(ERROR) << "input2->data_c() is nullptr";
      return RET_NULL_PTR;
    }
    auto ele_shape_data = reinterpret_cast<int *>(input2->data_c());
    for (int i = 0; i < input2->ElementsNum(); ++i) {
      element_shape_.push_back(ele_shape_data[i]);
    }
    auto status = MergeShape(input0->element_shape());
    if (status != RET_OK) {
      return RET_ERROR;
    }
    if (!IsFullyDefined(element_shape_)) {
      for (int i = 0; i < input0->ElementsNum(); ++i) {
        auto input = input0->GetTensor(i);
        MS_ASSERT(input != nullptr);
        if (input->data_type() != kTypeUnknown) {
          status = MergeShape(input->shape());
          if (status != RET_OK) {
            return RET_ERROR;
          }
        }
      }
    }
    if (!IsFullyDefined(element_shape_)) {
      MS_LOG(ERROR) << "element_shape_ is not fullyDefined!";
      return RET_ERROR;
    }
    output->set_data_type(input0->data_type());
    output->set_shape(element_shape_);
  }
  output->set_format(input0->GetTensor(index_)->format());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
