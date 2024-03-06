/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_TENSORLIST_H_
#define MINDSPORE_LITE_SRC_TENSORLIST_H_

#include <memory>
#include <vector>
#include "include/errorcode.h"
#include "nnacl/tensorlist_c.h"
#include "src/common/log_adapter.h"
#include "schema/model_generated.h"
#include "src/tensor.h"

namespace mindspore::lite {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
/**
 * Tensorlist is a container of vector, in which each element is a tensor object.
 * Member objects:
 *  1.tensors_: tensors_ is a vector, where each element is a pointer to tensor type.
 *  2.shape_: represents the size of the tensors_ and shape_.size() must be equal to 1.
 *  3.element_shape_: element_shape_ represents the shape of each tensor in tensors_.
 *    Some dimensions can be negative, which means that the corresponding dimensions of each tensor in tensors_ can be
 *    different.
 *  4.data_type_: indicates that the tensorlist is a tensor of type kObjectTypeTensorType, so it can only be
 *    "kObjectTypeTensorType"
 *  5.tensors_data_type_: data_type_ of each tensor in tensors_
 * Usage:
 *  std::vector<int> shape = (1, 2);  // tensors_ only has two tensor
 *  std::vector<int> element_shape = {-1, 99};
 *  // dim0 is arbitrary and dim1 is must to be 99 of each tensor.shape() in tensors_
 *  TensorList *tl = new TensorList(shape, element_shape);
 *  std::vector<std::vector<int> > tensor_shape = std::vector<vector<int> > (2,
 *                                                                          (std::vector<int> {5, 99},
 *                                                                           std::vector<int> {1, 99}));
 *  // tensor_shape[0] and tensor_shape[1] is not equal in dim0, but dim1 is must be equal to 99.
 *  t1->MallocTensorListData(kNumberTypeFloat, tensor_shape);
 *  t1->MallocData();
 *  t1->...
 *  ...
 *  t1->FreeData();
 *  t1->FreeTensorListData();
 *
 *  See the code for other constructors.
 */
class TensorList : public Tensor {
 public:
  TensorList() { tensor_list_c_ = {false, kObjectTypeTensorType, DEFAULT_FORMAT, 0, kTypeUnknown, -1, nullptr, 0, 0}; }

  TensorList(std::vector<int> shape, std::vector<int> element_shape, Category category = VAR);

  ~TensorList() override;

  TensorList(const TensorList &other) = delete;

  TensorList &operator=(const TensorList &tl) = delete;

  void set_element_shape(const std::vector<int> &shape) {
    if (shape.size() > MAX_SHAPE_SIZE) {
      FreeData();
      tensor_list_c_.element_shape_size_ = 0;
      MS_LOG(WARNING) << "The shape-size has exceeded the limit 8, now is " << shape.size();
      return;
    }
    tensor_list_c_.element_shape_size_ = shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
      tensor_list_c_.element_shape_[i] = shape[i];
    }
  }

  std::vector<int> element_shape() const {
    return std::vector<int>(tensor_list_c_.element_shape_,
                            tensor_list_c_.element_shape_ + tensor_list_c_.element_shape_size_);
  }

  void set_max_elements_num(int ele_num) { tensor_list_c_.max_elements_num_ = ele_num; }

  int max_elements_num() const { return tensor_list_c_.max_elements_num_; }

  static TensorList *CopyTensorList(const TensorList &src, bool copy_data = false,
                                    const AllocatorPtr &allocator = nullptr);

  int MallocTensorListData(TypeId dtype, const std::vector<std::vector<int> > &tensor_shape);

  int MallocData(const AllocatorPtr allocator = nullptr) override;

  int FreeTensorListData();

  void FreeData() override;

  int SetTensor(int index, const Tensor *src_tensor);

  Tensor *GetTensor(int index);

  void set_tensors_data_type(TypeId type) { tensor_list_c_.tensors_data_type_ = type; }

  TypeId tensors_data_type() const { return static_cast<TypeId>(tensor_list_c_.tensors_data_type_); }

  std::vector<Tensor *> tensors() { return tensors_; }

  void set_tensors(const std::vector<Tensor *> &tensors) { this->tensors_ = tensors; }

  int CheckTensorListParam();

  bool IsCompatibleShape(const std::vector<int> &shape);

  bool IsCompatibleShape(const Tensor *src);

  STATUS Decode(const int *data, size_t length);

  bool IsConst() const override;

  void set_init_ref_count(int ref_count) override {
    this->init_ref_count_ = ref_count;
    for (auto tensor : tensors_) {
      if (tensor != nullptr) {
        tensor->set_init_ref_count(ref_count);
      }
    }
  }

  void set_ref_count(int ref_count) override {
    ref_count_ = ref_count;
    for (auto tensor : tensors_) {
      if (tensor != nullptr) {
        tensor->set_ref_count(ref_count);
      }
    }
  }

  void ResetRefCount() override {
    set_ref_count(this->init_ref_count_);
    for (auto tensor : tensors_) {
      if (tensor != nullptr) {
        tensor->set_ref_count(this->init_ref_count_);
      }
    }
  }

  void IncRefCount() override {
    ++ref_count_;
    for (auto tensor : tensors_) {
      if (tensor != nullptr) {
        tensor->IncRefCount();
      }
    }
  }

  void DecRefCount() override {
    if (this->IsConst() || this->IsGraphInput()) {
      return;
    }
    --ref_count_;
    for (auto tensor : tensors_) {
      if (tensor != nullptr) {
        tensor->DecRefCount();
      }
    }
  }

  void set_allocator(AllocatorPtr allocator) override {
    allocator_ = allocator;
    for (auto tensor : tensors_) {
      if (tensor != nullptr) {
        tensor->set_allocator(allocator);
      }
    }
  }

  void set_own_data(bool own_data) override {
    this->own_data_ = own_data;
    for (auto tensor : tensors_) {
      if (tensor != nullptr) {
        tensor->set_own_data(own_data);
      }
    }
  }

  TensorListC *ConvertToTensorListC() {
    tensor_list_c_.format_ = tensor_c_.format_;
    tensor_list_c_.shape_value_ = tensor_c_.shape_size_ == 0 ? 0 : tensor_c_.shape_[0];
    tensor_list_c_.element_num_ = tensor_c_.shape_size_ == 0 ? 0 : tensors_.size();
    tensor_list_c_.tensors_ = nullptr;
    return &tensor_list_c_;
  }

 protected:
  // The following functions must be masked.
  void *data() const override { return nullptr; }
  void *MutableData() override { return nullptr; }
  size_t Size() const override { return 0; }
  TensorListC tensor_list_c_;
  std::vector<Tensor *> tensors_{};
};

#else

using TensorList = void;

#endif
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_TENSORLIST_H_
