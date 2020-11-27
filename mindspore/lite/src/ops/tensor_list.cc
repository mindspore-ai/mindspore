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
#include "src/ops/tensor_list.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

using mindspore::schema::Format_NC;

namespace mindspore {
namespace lite {

int TensorListFromTensor::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  // inputs0:tensor
  // inputs1: element_shape
  // outputs0: vector<tensor>.size() dtype
  // outputs1: element_shape
  // outputs2-n: vector<tensor>
  auto input = inputs_[0];
  MS_ASSERT(input != nullptr);
  std::vector<int> in_shape = input->shape();
  int dim0 = in_shape[0];
  if (dim0 <= 0) {
    MS_LOG(ERROR) << "inputs_[0] dim0:" << dim0 << " must greater than 0";
    return RET_ERROR;
  }
  std::vector<int> out_shape(in_shape.begin() + 1, in_shape.end());
  int out_vec_size = outputs_.size() - 2;
  if (out_vec_size != dim0) {
    MS_LOG(ERROR) << "outputs_.size() - 2:" << out_vec_size << "must be equal to dim0:" << dim0;
    return RET_ERROR;
  }
  for (int i = 0; i < dim0; ++i) {
    auto output = outputs_[i + 2];
    MS_ASSERT(output != nullptr);
    output->set_data_type(input->data_type());
    output->set_shape(out_shape);
  }

  auto output = outputs_[0];  // vector<tensor>.size(), tensorlist.dtype
  MS_ASSERT(output != nullptr);
  output->set_data_type(kNumberTypeInt);
  output->set_shape(std::vector<int>(1, 2));  // one element.value = 2

  output = outputs_[1];  // element_shape tensor
  MS_ASSERT(output != nullptr);
  output->set_data_type(inputs_[1]->data_type());
  output->set_format(inputs_[1]->format());
  output->set_shape(inputs_[1]->shape());

  return RET_OK;
}

bool TensorListGetItem::IsFullyDefined(const std::vector<int> &shape) const {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      return false;
    }
  }
  return true;
}

int TensorListGetItem::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  int in_vec_size = inputs_.size();
  auto input0 = inputs_[0];
  MS_ASSERT(input0 != nullptr);
  auto in0_ptr = reinterpret_cast<int *>(input0->data_c());
  if (in_vec_size != in0_ptr[0] + 4) {
    MS_LOG(ERROR) << "inputs_.size():" << in_vec_size << " must be equal to:" << in0_ptr[0] + 4;
    return RET_ERROR;
  }
  auto get_index = inputs_[in0_ptr[0] + 2];
  MS_ASSERT(get_index != nullptr);
  index_ = reinterpret_cast<int *>(get_index->data_c())[0];
  if (index_ < 0 || index_ > in0_ptr[0]) {
    MS_LOG(ERROR) << "index_:" << index_ << "must in [0, " << in0_ptr[0] << "]";
    return RET_ERROR;
  }
  auto input_index = inputs_[index_ + 2];
  MS_ASSERT(input_index != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (input_index->data_type() != kTypeUnknown) {
    output->set_format(input_index->format());
    output->set_data_type(input_index->data_type());
    output->set_shape(input_index->shape());
  } else {
    auto ele_shape_tensor = inputs_[in0_ptr[0] + 3];
    MS_ASSERT(ele_shape_tensor != nullptr);
    auto ele_shape_type = ele_shape_tensor->data_type();
    if (ele_shape_type != kNumberTypeInt) {
      MS_LOG(ERROR) << "ele_shape_tensor.data_type():" << ele_shape_type
                    << " must be \"kNumberTypeInt\":" << kNumberTypeInt;
      return RET_ERROR;
    }
    auto shape_ptr = reinterpret_cast<int *>(ele_shape_tensor->data_c());
    for (int i = 0; i < ele_shape_tensor->ElementsNum(); ++i) {
      element_shape_.push_back(shape_ptr[i]);
    }
    if (!IsFullyDefined(element_shape_)) {
      for (int i = 0; i < in0_ptr[0]; ++i) {
        auto input = inputs_[i + 2];
        if (input->data_type() != kTypeUnknown) {
          std::vector<int> tmp = input->shape();
          for (size_t j = 0; j < tmp.size(); ++j) {
            element_shape_[j] = element_shape_[j] >= 0 ? element_shape_[j] : tmp[j];
          }
        }
      }
    }
    if (!IsFullyDefined(element_shape_)) {
      MS_LOG(ERROR) << "ele_shape_tensor Is Not FullyDefined!";
      return RET_ERROR;
    }
    element_dtype_ = GetElementDType();
    output->set_data_type(element_dtype_);
    output->set_shape(element_shape_);
  }
  return RET_OK;
}
#ifdef PRIMITIVE_WRITEABLE
TypeId TensorListFromTensor::GetElementDType() const {
  return (TypeId)(this->primitive_->value.AsTensorListFromTensor()->elementDType);
}

TypeId TensorListFromTensor::GetShapeType() const {
  return (TypeId)(this->primitive_->value.AsTensorListFromTensor()->shapeType);
}

TypeId TensorListGetItem::GetElementDType() const {
  return (TypeId)(this->primitive_->value.AsTensorListGetItem()->elementDType);
}
TypeId TensorListReserve::GetElementDType() const {
  return (TypeId)(this->primitive_->value.AsTensorListReserve()->elementDType);
}

TypeId TensorListStack::GetElementDType() const {
  return (TypeId)(this->primitive_->value.AsTensorListStack()->elementDType);
}

int TensorListStack::GetNumElements() const { return this->primitive_->value.AsTensorListStack()->numElements; }

#else
TypeId TensorListFromTensor::GetElementDType() const {
  return (TypeId)(this->primitive_->value_as_TensorListFromTensor()->elementDType());
}

TypeId TensorListFromTensor::GetShapeType() const {
  return (TypeId)(this->primitive_->value_as_TensorListFromTensor()->shapeType());
}

TypeId TensorListGetItem::GetElementDType() const {
  return (TypeId)(this->primitive_->value_as_TensorListGetItem()->elementDType());
}
TypeId TensorListReserve::GetElementDType() const {
  return (TypeId)(this->primitive_->value_as_TensorListReserve()->elementDType());
}

TypeId TensorListStack::GetElementDType() const {
  return (TypeId)(this->primitive_->value_as_TensorListStack()->elementDType());
}

int TensorListStack::GetNumElements() const { return this->primitive_->value_as_TensorListStack()->numElements(); }
#endif

int TensorListReserve::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  // input0: element_shape_tensor
  // input1: num_elements
  auto input0 = inputs_.front();
  MS_ASSERT(input0 != nullptr);
  auto ele_shape_type = input0->data_type();
  if (ele_shape_type != kNumberTypeInt) {
    MS_LOG(ERROR) << "ele_shape_tensor.data_type():" << ele_shape_type
                  << " must be \"kNumberTypeInt\":" << kNumberTypeInt;
    return RET_ERROR;
  }
  auto input1 = inputs_[1];
  MS_ASSERT(input1 != nullptr);
  auto num_ele_type = input1->data_type();
  if (num_ele_type != kNumberTypeInt) {
    MS_LOG(ERROR) << "num_ele_tensor.data_type():" << num_ele_type << " must be \"kNumberTypeInt\":" << kNumberTypeInt;
    return RET_ERROR;
  }
  int num_elements = reinterpret_cast<int *>(input1->data_c())[0];
  auto out_vec_size = outputs_.size();
  if (out_vec_size != (size_t)(num_elements + 2)) {
    MS_LOG(ERROR) << "outputs_.size():" << out_vec_size << " must be equal to:" << num_elements + 2;
    return RET_ERROR;
  }

  for (int i = 0; i < num_elements; ++i) {
    auto output = outputs_[i + 2];
    MS_ASSERT(output != nullptr);
    output->set_data_type(kTypeUnknown);
    output->set_shape(std::vector<int>(1, 0));  // shape = [0]
  }

  auto output = outputs_[0];  // vector<tensor>.size(), tensorlist.dtype
  MS_ASSERT(output != nullptr);
  output->set_data_type(kNumberTypeInt);
  output->set_shape(std::vector<int>(1, 2));  // one element.value = 2

  output = outputs_[1];  // element_shape tensor
  MS_ASSERT(output != nullptr);
  output->set_data_type(input0->data_type());
  output->set_format(input0->format());
  output->set_shape(input0->shape());
  return RET_OK;
}

bool TensorListStack::IsFullyDefined(const std::vector<int> &shape) const {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      return false;
    }
  }
  return true;
}

int TensorListStack::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  // input0: tensorlist
  // input[inputs_.size() - 1]: element_shape
  auto input0 = inputs_.front();
  MS_ASSERT(input0 != nullptr);
  auto input0_ptr = reinterpret_cast<int *>(input0->data_c());
  int vec_in_size = inputs_.size();
  if (vec_in_size != input0_ptr[0] + 3) {
    MS_LOG(ERROR) << "inputs_.size():" << vec_in_size << " must be equal:" << input0_ptr[0] + 3;
    return RET_ERROR;
  }
  auto ele_shape = inputs_[input0_ptr[0] + 2];  // element shape
  MS_ASSERT(ele_shape != nullptr);
  auto ele_shape_ptr = reinterpret_cast<int *>(ele_shape->data_c());
  for (int i = 0; ele_shape->ElementsNum(); ++i) {
    output_shape_.push_back(ele_shape_ptr[i]);
  }
  std::vector<int> tensorlist_shape;
  MS_ASSERT(inputs_[1] != nullptr);
  auto input1_ptr = reinterpret_cast<int *>(inputs_[1]->data_c());
  for (int i = 0; i < inputs_[1]->ElementsNum(); ++i) {
    tensorlist_shape.push_back(input1_ptr[i]);
  }
  auto status = MergeShape(tensorlist_shape);
  if (status == RET_ERROR) {
    MS_LOG(ERROR) << "Merge tensorlist_shape is error!";
    return RET_ERROR;
  }
  if (!IsFullyDefined(output_shape_)) {
    MS_LOG(ERROR) << "element_shape Is Not FullyDefined!";
    return RET_ERROR;
  }
  if (!IsFullyDefined(tensorlist_shape)) {
    for (int i = 0; i < input0_ptr[0]; ++i) {  // get tensorlist every tensor
      auto tensor_tmp = inputs_[i + 2];
      MS_ASSERT(tensor_tmp != nullptr);
      if (tensor_tmp->data_type() != kTypeUnknown) {
        status = MergeShape(tensor_tmp->shape());
        if (status == RET_ERROR) {
          MS_LOG(ERROR) << "Merge inputs_[" << i + 2 << "] is error!";
          return RET_ERROR;
        }
      }
    }
  }
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_format(Format_NC);
  output->set_data_type(static_cast<TypeId>(input0_ptr[1]));
  output->set_shape(std::vector<int>(
    1, input0_ptr[0] * std::accumulate(output_shape_.begin(), output_shape_.end(), 1LL, std::multiplies<int>())));
  return RET_OK;
}

int TensorListStack::MergeShape(const std::vector<int> &shape) {
  size_t dim0 = shape.size();
  size_t dim1 = output_shape_.size();
  if (dim1 >= unKnownRank_) {
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
    int tmp_size = dim1_size >= 0 ? dim1_size : dim0_size;
    output_shape_[i] = tmp_size;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
