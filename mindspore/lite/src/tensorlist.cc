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
#include "include/ms_tensor.h"
#include "src/common/log_adapter.h"
#include "schema/model_generated.h"
#include "src/tensor.h"

namespace mindspore::lite {

TensorList::TensorList(std::vector<int> shape, std::vector<int> element_shape, Category category)
    : Tensor(kObjectTypeTensorType, std::move(shape), schema::Format::Format_NHWC, category),
      element_shape_(std::move(element_shape)) {}

TensorList::~TensorList() {
  if (!this->tensors_.empty()) {
    this->TensorList::FreeData();
    this->FreeTensorListData();
  }
}

int TensorList::CopyTensorList(const TensorList &src, bool copy_data) {
  this->data_type_ = src.data_type_;
  this->tensors_data_type_ = src.tensors_data_type_;
  this->shape_ = src.shape_;
  this->element_shape_ = src.element_shape_;
  this->max_elements_num_ = src.max_elements_num_;
  if (copy_data) {
    auto ret = CopyTensorData(src);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CopyTensorData error";
      return RET_ERROR;
    }
  } else {
    for (auto tensor : this->tensors()) {
      delete tensor;
    }
    this->tensors_.clear();
    // each tensor in tensors_ will share the same memory space.
    this->tensors_ = src.tensors_;
  }
  return RET_OK;
}

int TensorList::CopyTensorData(const TensorList &src) {
  if (src.tensors_.empty()) {
    return RET_OK;
  }
  for (auto tensor : this->tensors()) {
    delete tensor;
  }
  this->tensors_.clear();
  for (int i = 0; i < this->ElementsNum(); ++i) {
    if (src.tensors_[i] == nullptr) {
      MS_LOG(ERROR) << "src tensors_[" << i << "] is nullptr!";
      return RET_ERROR;
    }
    auto dst_tensor = Tensor::CopyTensor(*src.tensors_[i]);
    if (dst_tensor == nullptr) {
      MS_LOG(ERROR) << "CopyTensorData: new tensor[" << i << "] is failed!";
      return RET_ERROR;
    }
    this->tensors_.push_back(dst_tensor);
  }
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
  if (this->shape().size() != 1) {
    MS_LOG(ERROR) << "tensorlist shape:" << this->shape().size() << " must be one-dimensional";
    return RET_ERROR;
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
    this->tensors_.push_back(tensor_ptr);
  }
  return RET_OK;
}

int TensorList::MallocData(const mindspore::Allocator *allocator) {
  if (allocator != nullptr) {
    allocator_ = const_cast<mindspore::Allocator *>(allocator);
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

void TensorList::FreeData() {
  // free data buf of each tensor in tensors_
  for (auto tensor : tensors_) {
    tensor->FreeData();
  }
}

int TensorList::FreeTensorListData() {
  // del each tensor in tensors_ and clear tensors_
  if (this->tensors_.empty()) {
    return RET_OK;
  }
  for (int i = 0; i < this->ElementsNum(); ++i) {
    if (this->tensors_[i] != nullptr) {
      delete this->tensors_[i];
      this->tensors_[i] = nullptr;
    }
  }
  tensors_.clear();
  return RET_OK;
}

int TensorList::SetTensor(int index, Tensor *src_tensor) {
  // your can use this fun to modify tensor[index] value
  if (src_tensor->data_type() != this->tensors_data_type_) {
    MS_LOG(ERROR) << "src_tensor->data_type()：" << src_tensor->data_type()
                  << " must be equal to tensors_data_type_:" << this->tensors_data_type_;
    return RET_ERROR;
  }
  if (index < 0 || index > (this->ElementsNum() - 1)) {
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

int TensorList::set_root_tensor(Tensor *tensor) {
  auto ret = Tensor::set_root_tensor(tensor);
  if (ret != RET_OK) {
    return ret;
  }
  if (this->data_type_ != kObjectTypeTensorType) {
    return RET_OK;
  }
  auto root_tensorlist = reinterpret_cast<TensorList *>(this->root_tensor_);
  if (root_tensorlist == nullptr) {
    MS_LOG(ERROR) << "root_tensor of tensorlist should be a tensorlist";
    return RET_INFER_INVALID;
  }
  this->element_shape_ = root_tensorlist->element_shape_;
  this->max_elements_num_ = root_tensorlist->max_elements_num_;
  this->tensors_data_type_ = root_tensorlist->tensors_data_type_;
  return RET_OK;
}

Tensor *TensorList::GetTensor(int index) {
  // return tensor[index] ptr. With this function, you can modify tensors_[index] at will.
  if (this->root_tensor_ != nullptr) {
    if (this->data_type_ != kObjectTypeTensorType) {
      MS_LOG(ERROR) << "root_tensor of tensorlist should be a tensorlist";
      return nullptr;
    }
    auto root_tensorlist = reinterpret_cast<TensorList *>(this->root_tensor_);
    if (index < 0 || index >= static_cast<int>(root_tensorlist->tensors_.size())) {
      MS_LOG(ERROR) << "index:" << index << " must in [0, " << this->ElementsNum() - 1 << "]!";
      return nullptr;
    }
    return root_tensorlist->tensors_[index];
  }
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
  // shape is store in Tensor.
  if (static_cast<size_t>(src->ElementsNum()) != this->element_shape_.size()) {
    return false;
  }
  if (src->data_type() != kNumberTypeInt && src->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "src tensor data_type:" << src->data_type() << " is not int";
    return false;
  }
  auto src_ptr = reinterpret_cast<int *>(src->data_c());
  for (size_t i = 0; i < this->element_shape_.size(); ++i) {
    if (this->element_shape_[i] >= 0 && src_ptr[i] >= 0 && this->element_shape_[i] != src_ptr[i]) {
      return false;
    }
  }
  return true;
}

STATUS TensorList::Decode(const int *data) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "data is nullptr";
    return RET_ERROR;
  }
  tensors_data_type_ = TypeId(data[0]);
  for (int j = 0; j < data[1]; ++j) {
    element_shape_.push_back(data[2 + j]);
  }
  int tensors_num = data[2 + data[1]];
  tensors_.resize(tensors_num);
  int tensor_index = 2 + data[1] + 1;
  for (int i = 0; i < tensors_num; i++) {
    int tensor_dims_size = data[tensor_index++];
    std::vector<int> shape(tensor_dims_size);
    for (int j = 0; j < tensor_dims_size; j++) {
      shape[j] = data[tensor_index++];
    }
    tensors_[i] = new (std::nothrow) Tensor(tensors_data_type_, shape);
    if (tensors_[i] == nullptr) {
      MS_LOG(ERROR) << "new Tensor failed";
      return RET_NULL_PTR;
    }
  }
  return RET_OK;
}

bool TensorList::IsConst() const { return this->category_ == CONST_TENSOR || this->category_ == CONST_SCALAR; }

}  // namespace mindspore::lite
