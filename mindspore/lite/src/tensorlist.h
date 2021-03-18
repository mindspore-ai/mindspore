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

#ifndef MINDSPORE_LITE_SRC_TENSORLIST_H_
#define MINDSPORE_LITE_SRC_TENSORLIST_H_

#include <memory>
#include <vector>
#include "include/ms_tensor.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "schema/model_generated.h"
#include "src/tensor.h"

namespace mindspore::lite {
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
  TensorList() = default;

  TensorList(std::vector<int> shape, std::vector<int> element_shape, Category category = VAR);

  ~TensorList() override;

  TensorList(const TensorList &other) = delete;

  TensorList &operator=(const TensorList &tl) = delete;

  void set_element_shape(const std::vector<int> &shape) { element_shape_ = shape; }

  std::vector<int> &element_shape() { return element_shape_; }

  void set_max_elements_num(int ele_num) { max_elements_num_ = ele_num; }

  int max_elements_num() const { return max_elements_num_; }

  int MallocTensorListData(TypeId dtype, const std::vector<std::vector<int> > &tensor_shape);

  int MallocData(const mindspore::Allocator *allocator = nullptr) override;

  int FreeTensorListData();

  void FreeData() override;

  int CopyTensorList(const TensorList &src, bool copy_data);

  int CopyTensorData(const TensorList &src);

  int SetTensor(int index, Tensor *src_tensor);

  Tensor *GetTensor(int index);

  void set_tensors_data_type(TypeId type) { tensors_data_type_ = type; }

  TypeId tensors_data_type() const { return tensors_data_type_; }

  std::vector<Tensor *> &tensors() { return tensors_; }

  void set_tensors(const std::vector<Tensor *> &tensors) { this->tensors_ = tensors; }

  int CheckTensorListParam();

  bool IsCompatibleShape(const std::vector<int> &shape);

  bool IsCompatibleShape(const Tensor *src);

  STATUS Decode(const int *data);

  bool IsConst() const override;

  int set_root_tensor(Tensor *tensor) override;

 protected:
  // The following functions must be masked.
  void set_data(void *data) override {}
  void *data_c() const override { return nullptr; }
  void *MutableData() override { return nullptr; }
  size_t Size() const override { return 0; }
  std::vector<Tensor *> tensors_{};
  TypeId tensors_data_type_ = kTypeUnknown;
  std::vector<int> element_shape_{};
  int max_elements_num_ = -1;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_TENSORLIST_H_
