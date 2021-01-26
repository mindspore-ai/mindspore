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
#include "src/ops/tensorlist_stack.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
TypeId TensorListStack::GetElementDType() const {
  return (TypeId)(this->primitive_->value.AsTensorListStack()->elementDType);
}

int TensorListStack::GetNumElements() const { return this->primitive_->value.AsTensorListStack()->numElements; }

void TensorListStack::SetElementDType(int type) { this->primitive_->value.AsTensorListStack()->elementDType = type; }

void TensorListStack::SetNumElements(int num_elements) {
  this->primitive_->value.AsTensorListStack()->numElements = num_elements;
}

int TensorListStack::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_TensorListStack;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_TensorListStack) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::TensorListStackT();
    if (attr == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      MS_LOG(ERROR) << "new TensorListStackT value failed";
      return RET_ERROR;
    }
    if (prim.GetAttr("elementDType") == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      delete attr;
      MS_LOG(ERROR) << "TensorListStack's attr elementDType is not set";
      return RET_ERROR;
    } else {
      attr->elementDType = CastToInt(prim.GetAttr("elementDType")).front();
    }
    if (prim.GetAttr("numElements") == nullptr) {
      delete this->primitive_;
      this->primitive_ = nullptr;
      delete attr;
      MS_LOG(ERROR) << "TensorListStack's attr numElements is not set";
      return RET_ERROR;
    } else {
      attr->numElements = CastToInt(prim.GetAttr("numElements")).front();
    }
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}
#else
TypeId TensorListStack::GetElementDType() const {
  return (TypeId)(this->primitive_->value_as_TensorListStack()->elementDType());
}

int TensorListStack::GetNumElements() const { return this->primitive_->value_as_TensorListStack()->numElements(); }

int TensorListStack::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_TensorListStack();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_TensorListStack return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateTensorListStack(*fbb, attr->numElements(), attr->elementDType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_TensorListStack, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *TensorListStackCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<TensorListStack>(primitive);
}
Registry TensorListStackRegistry(schema::PrimitiveType_TensorListStack, TensorListStackCreator);
#endif

bool TensorListStack::IsFullyDefined(const std::vector<int> &shape) const {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      return false;
    }
  }
  return true;
}

int TensorListStack::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto input0 = reinterpret_cast<TensorList *>(inputs_.front());
  MS_ASSERT(input0 != nullptr);
  if (input0->ElementsNum() == 0) {
    MS_LOG(ERROR) << "Try to stack a empty tensorlist!";
    return RET_ERROR;
  }
  auto ele_shape = inputs_[1];  // element shape
  MS_ASSERT(ele_shape != nullptr);
  if (ele_shape->data_c() == nullptr) {
    MS_LOG(ERROR) << "ele_shape->data_c() is nullptr";
    return RET_NULL_PTR;
  }
  auto ele_shape_ptr = reinterpret_cast<int *>(ele_shape->data_c());
  output_shape_.clear();
  for (int i = 0; i < ele_shape->ElementsNum(); ++i) {
    output_shape_.push_back(ele_shape_ptr[i]);
  }

  auto status = MergeShape(input0->element_shape());
  if (status == RET_ERROR) {
    MS_LOG(ERROR) << "Merge element_shape is error!";
    return RET_ERROR;
  }
  if (!IsFullyDefined(output_shape_)) {
    MS_LOG(ERROR) << "output_shape_ Is Not FullyDefined!";
    return RET_ERROR;
  }
  if (!IsFullyDefined(input0->element_shape())) {
    for (int i = 0; i < input0->ElementsNum(); ++i) {
      auto tensor_ele = input0->GetTensor(i);
      MS_ASSERT(tensor_ele != nullptr);
      if (tensor_ele->data_type() != kTypeUnknown) {
        status = MergeShape(tensor_ele->shape());
        if (status == RET_ERROR) {
          MS_LOG(ERROR) << "Merge input0->tensors_[" << i << "] is error!";
          return RET_ERROR;
        }
      }
    }
  }
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(input0->tensors_data_type());
  output_shape_.insert(output_shape_.begin(), input0->ElementsNum());
  output->set_shape(output_shape_);
  output->set_format(input0->format());
  return RET_OK;
}

int TensorListStack::MergeShape(const std::vector<int> &shape) {
  size_t dim0 = shape.size();
  size_t dim1 = output_shape_.size();
  if (dim1 >= unKnownRank_ || output_shape_[0] == -1) {
    output_shape_ = shape;
    return RET_OK;
  }
  if (dim1 != dim0) {
    MS_LOG(ERROR) << "shape.size():" << dim1 << " must be equal output_shape_.size():" << dim0;
    return RET_ERROR;
  }
  for (size_t i = 0; i < dim0; ++i) {
    int dim0_size = shape[i];
    int dim1_size = output_shape_[i];
    if (dim0_size >= 0 && dim1_size >= 0 && dim0_size != dim1_size) {
      MS_LOG(ERROR) << "shape[" << i << "]:" << dim0_size << " is incompatible with output_shape_[" << i
                    << "]:" << dim1_size;
      return RET_ERROR;
    }
    output_shape_[i] = dim1_size >= 0 ? dim1_size : dim0_size;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
