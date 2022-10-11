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

#include "src/tensorlist.h"
#include <utility>
#include <algorithm>
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
TensorList::TensorList(std::vector<int> shape, std::vector<int> element_shape, Category category)
    : Tensor(kObjectTypeTensorType, std::move(shape), mindspore::NHWC, category),
      element_shape_(std::move(element_shape)) {}

TensorList::~TensorList() {
  if (!this->tensors_.empty()) {
    this->TensorList::FreeData();
    this->FreeTensorListData();
  }
}

void TensorList::FreeData() {
  if (this->IsConst() || this->IsGraphInput()) {
    return;
  }
  // free data buf of each tensor in tensors_
  for (auto tensor : tensors_) {
    if (tensor == nullptr) {
      continue;
    }
    tensor->FreeData();
  }
}

int TensorList::FreeTensorListData() {
  // del each tensor in tensors_ and clear tensors_
  if (this->tensors_.empty()) {
    return RET_OK;
  }
  for (auto &tensor : this->tensors_) {
    if (tensor != nullptr) {
      delete tensor;
      tensor = nullptr;
    }
  }
  tensors_.clear();
  return RET_OK;
}

int TensorList::MallocTensorListData(TypeId dtype, const std::vector<std::vector<int> > &tensor_shape) {
  // This function will create a new tensors_
  // Your must to set shape(param2: tensor_shape) and data_type_(tensors_data_type_ = param1: dtype) of each tensor in
  // tensors_. After that, you need to call function:MallocData to malloc data buf of each tensor in tensors_.
  if (!this->tensors_.empty()) {
    // If tensors_ is not empty then clear this tensors_ and rebuild a new tensors_.
    auto ret = FreeTensorListData();
    if (ret != RET_OK) {
      return RET_ERROR;
    }
  }
  if (this->shape().size() == 0) {
    MS_LOG(INFO) << "tensorlist has no elements, no need malloc data.";
    return RET_OK;
  }
  if (this->shape().size() != 1) {
    MS_LOG(ERROR) << "tensorlist shape:" << this->shape().size() << " must be one-dimensional";
    return RET_ERROR;
  }
  if (tensor_shape.empty()) {
    MS_LOG(INFO) << "tensor_shape is empty, no need malloc tensor list data";
    return RET_OK;
  }
  if (static_cast<size_t>(this->ElementsNum()) != tensor_shape.size()) {
    MS_LOG(ERROR) << "tensorlist ElementsNum():" << this->ElementsNum()
                  << " must be equal to param2:tensor_shape.size():" << tensor_shape.size();
    return RET_ERROR;
  }
  this->tensors_data_type_ = dtype;
  for (int i = 0; i < this->ElementsNum(); ++i) {
    auto tensor_ptr = new (std::nothrow) Tensor(dtype, tensor_shape[i]);
    if (tensor_ptr == nullptr) {
      MS_LOG(ERROR) << "new tensors_[" << i << "] is failed!";
      return RET_ERROR;
    }
    if (!this->allocator()) {
      tensor_ptr->set_allocator(this->allocator());
    }
    tensor_ptr->set_init_ref_count(this->init_ref_count());
    tensor_ptr->set_ref_count(this->ref_count());
    this->tensors_.push_back(tensor_ptr);
  }
  return RET_OK;
}

int TensorList::MallocData(const AllocatorPtr allocator) {
  if (allocator != nullptr) {
    allocator_ = allocator;
  }
  // malloc data buf of each tensor in tensors_
  for (int i = 0; i < this->ElementsNum(); ++i) {
    if (tensors_.empty()) {
      return RET_OK;
    }
    auto tensor_ptr = this->tensors_[i];
    if (tensor_ptr == nullptr) {
      MS_LOG(ERROR) << "tensors_[" << i << "] is nullptr!";
      return RET_ERROR;
    }
    // if data_type() is kTypeUnknown then data buf will not to be malloc
    if (tensor_ptr->data_type() != kTypeUnknown) {
      auto ret = tensor_ptr->MallocData(this->allocator_);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "tensorlist malloc tensors_[:" << i << "] is failed!";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int TensorList::SetTensor(int index, const Tensor *src_tensor) {
  MS_CHECK_TRUE_MSG(src_tensor != nullptr, RET_ERROR, "src tensor cannot null");
  // your can use this fun to modify tensor[index] value
  if (src_tensor->data_type() != this->tensors_data_type_) {
    MS_LOG(ERROR) << "src_tensor->data_type()：" << src_tensor->data_type()
                  << " must be equal to tensors_data_type_:" << this->tensors_data_type_;
    return RET_ERROR;
  }
  auto element_num = this->ElementsNum();
  MS_CHECK_GE(element_num, 0, RET_ERROR);
  if (index < 0 || index > (element_num - 1)) {
    MS_LOG(ERROR) << "index:" << index << " must in [0, " << this->ElementsNum() - 1 << "]!";
    return RET_ERROR;
  }
  auto dst_tensor = this->tensors_[index];
  // free original tensor data
  delete dst_tensor;
  this->tensors_[index] = Tensor::CopyTensor(*src_tensor);
  if (this->tensors_[index] == nullptr) {
    MS_LOG(ERROR) << "SetTensor: new tensor is failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

int TensorList::CheckTensorListParam() {
  for (int i = 0; i < this->ElementsNum(); ++i) {
    // each tensor in tensorlist must be not nullptr
    if (this->tensors_[i] == nullptr) {
      MS_LOG(ERROR) << "CheckTensorListParam: tensors_[" << i << "] is nullptr";
      return RET_ERROR;
    }
    if (this->tensors_[i]->data_type() != this->tensors_data_type_) {
      MS_LOG(ERROR) << "CheckTensorListParam: tensors_[i] data_type:" << this->tensors_[i]->data_type()
                    << " is not equal to tensors_data_type_:" << this->tensors_data_type_;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

Tensor *TensorList::GetTensor(int index) {
  // return tensor[index] ptr. With this function, you can modify tensors_[index] at will.
  if (index < 0 || index >= static_cast<int>(this->tensors_.size())) {
    MS_LOG(ERROR) << "index:" << index << " must in [0, " << this->ElementsNum() - 1 << "]!";
    return nullptr;
  }
  return this->tensors_[index];
}

bool TensorList::IsCompatibleShape(const std::vector<int> &shape) {
  if (this->tensors_.empty() && this->element_shape_.empty()) {
    return true;
  }
  if (shape.size() != this->element_shape_.size()) {
    return false;
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (this->element_shape_[i] >= 0 && shape[i] >= 0 && this->element_shape_[i] != shape[i]) {
      return false;
    }
  }
  return true;
}

bool TensorList::IsCompatibleShape(const Tensor *src) {
  MS_CHECK_TRUE_MSG(src != nullptr, false, "src tensor cannot null");
  // shape is store in Tensor.
  if (static_cast<size_t>(src->ElementsNum()) != this->element_shape_.size()) {
    return false;
  }
  if (src->data_type() != kNumberTypeInt && src->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "src tensor data_type:" << src->data_type() << " is not int";
    return false;
  }
  auto src_ptr = reinterpret_cast<int *>(src->data());
  for (size_t i = 0; i < this->element_shape_.size(); ++i) {
    if (this->element_shape_[i] >= 0 && src_ptr[i] >= 0 && this->element_shape_[i] != src_ptr[i]) {
      return false;
    }
  }
  return true;
}

STATUS TensorList::Decode(const int *data, size_t length) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "data is nullptr";
    return RET_ERROR;
  }
  MS_CHECK_LT(1, length, RET_ERROR);
  tensors_data_type_ = TypeId(data[0]);
  if (tensors_data_type_ < kTypeUnknown || tensors_data_type_ > kMonadTypeEnd) {
    MS_LOG(ERROR) << "TypeId illegal.";
    return RET_ERROR;
  }
  if (data[1] < 0) {
    MS_LOG(ERROR) << "element shape size illegal.";
    return RET_ERROR;
  }
  constexpr int kShapeIndexStart = 2;
  MS_CHECK_LT(static_cast<size_t>(data[1] + kShapeIndexStart), length, RET_ERROR);
  for (int j = 0; j < data[1]; ++j) {
    element_shape_.push_back(data[2 + j]);
  }
  int tensors_num = data[2 + data[1]];
  if (tensors_num < 0) {
    MS_LOG(WARNING) << "not able to create tensors, need infer shape.";
    return RET_OK;
  }

  if (this->ElementsNum() != tensors_num) {
    MS_LOG(WARNING) << "Input tensorlist data is invalid: shape size(" << this->ElementsNum() << ") != tensors_num("
                    << tensors_num << ").";
    MS_LOG(WARNING) << "tensor name: " << this->tensor_name_;
  }
  tensors_.reserve(tensors_num);
  int tensor_index = 2 + data[1] + 1;
  for (int i = 0; i < tensors_num; i++) {
    MS_CHECK_LT(static_cast<size_t>(tensor_index), length, RET_ERROR);
    int tensor_dims_size = data[tensor_index++];
    std::vector<int> shape(tensor_dims_size);
    for (int j = 0; j < tensor_dims_size; j++) {
      MS_CHECK_LT(static_cast<size_t>(tensor_index), length, RET_ERROR);
      shape[j] = data[tensor_index++];
    }
    auto tensor = new (std::nothrow) Tensor(tensors_data_type_, shape);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "new Tensor failed";
      return RET_NULL_PTR;
    }
    tensors_.emplace_back(tensor);
  }
  return RET_OK;
}

bool TensorList::IsConst() const { return this->category_ == CONST_TENSOR || this->category_ == CONST_SCALAR; }

TensorList *TensorList::CopyTensorList(const TensorList &src, bool copy_data, AllocatorPtr allocator) {
  auto *result = new TensorList;
  if (result == nullptr) {
    MS_LOG(ERROR) << "New tensor failed";
    return nullptr;
  }
  result->data_type_ = src.data_type_;
  result->shape_ = src.shape_;
  result->category_ = src.category_;
  result->format_ = src.format_;
  result->tensors_data_type_ = src.tensors_data_type_;
  result->element_shape_ = src.element_shape_;
  result->set_allocator(allocator);
  result->set_tensor_name(src.tensor_name() + "_duplicate");
  auto src_tensor_dtype = src.tensors_data_type_;
  std::vector<std::vector<int> > tensor_shape{};
  std::transform(src.tensors_.begin(), src.tensors_.end(), std::back_inserter(tensor_shape),
                 [](const Tensor *tensor_item) { return tensor_item->shape(); });

  for (LiteQuantParam quant : src.quant_params()) {
    result->AddQuantParam(quant);
  }

  if (result->shape().empty()) {
    return result;
  }
  result->MallocTensorListData(src_tensor_dtype, tensor_shape);
  if (copy_data) {
    for (size_t i = 1; i < src.tensors_.size(); ++i) {
      auto src_tensor = src.tensors_;
      auto ret = Tensor::CopyTensorData(*(src.tensors_[i]), result->tensors_[i]);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "CopyTensorData error";
        delete result;
        return nullptr;
      }
    }
    result->own_data_ = src.own_data_;
  }

  return result;
}
#endif
}  // namespace mindspore::lite
