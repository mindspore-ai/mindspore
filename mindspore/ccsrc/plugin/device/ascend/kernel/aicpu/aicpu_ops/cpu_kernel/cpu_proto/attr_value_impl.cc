/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cpu_kernel/cpu_proto/attr_value_impl.h"

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "cpu_kernel/cpu_proto/tensor_impl.h"
#include "cpu_kernel/cpu_proto/tensor_shape_impl.h"

namespace aicpu {
/*
 * get string value of attr.
 */
std::string AttrValueImpl::GetString() const { return attr_value_->s(); }

/*
 * get string list size of attr.
 */
int32_t AttrValueImpl::ListStringSize() const {
  auto array = attr_value_->array();
  return array.s_size();
}

/*
 * get string list value of attr.
 */
std::vector<std::string> AttrValueImpl::GetListString() const {
  std::vector<std::string> ret;
  auto array = attr_value_->array();
  for (int32_t i = 0; i < array.s_size(); i++) {
    ret.emplace_back(array.s(i));
  }
  return ret;
}

/*
 * set string list value to attr.
 */
void AttrValueImpl::SetListString(const std::vector<std::string> &bytes) {
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
  for (const std::string &s : bytes) {
    array->add_s(s);
  }
}

/*
 * set string value to attr.
 */
void AttrValueImpl::SetString(const std::string &byte) { attr_value_->set_s(byte); }

/*
 * attr add string value to list.
 */
void AttrValueImpl::AddListString(const std::string &str) {
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
  array->add_s(str);
}

/*
 * get int value of attr.
 */
int64_t AttrValueImpl::GetInt() const { return attr_value_->i(); }

/*
 * get int list value of attr.
 */
std::vector<int64_t> AttrValueImpl::GetListInt() const {
  std::vector<int64_t> ret;
  auto array = attr_value_->array();
  for (int32_t i = 0; i < array.i_size(); i++) {
    ret.emplace_back(array.i(i));
  }
  return ret;
}

/*
 * attr add int value to list.
 */
void AttrValueImpl::AddListInt(int64_t i) {
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
  array->add_i(i);
}

/*
 * get int list size of attr.
 */
int32_t AttrValueImpl::ListIntSize() const {
  auto array = attr_value_->array();
  return array.i_size();
}

/*
 * set int value to attr.
 */
void AttrValueImpl::SetInt(int64_t i) { attr_value_->set_i(i); }

/*
 * set int list value to attr.
 */
void AttrValueImpl::SetListInt(const std::vector<int64_t> &list) {
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
  for (const int64_t &i : list) {
    array->add_i(i);
  }
}

/*
 * get int list list value of attr.
 */
std::vector<std::vector<int64_t>> AttrValueImpl::GetListListInt() const {
  auto array = attr_value_->list_list_int();
  std::vector<std::vector<int64_t>> ret;
  for (auto idx = 0; idx < array.list_list_i_size(); ++idx) {
    std::vector<int64_t> vec;
    for (auto i = 0; i < array.list_list_i(idx).list_i_size(); ++i) {
      vec.emplace_back(array.list_list_i(idx).list_i(i));
    }
    ret.emplace_back(vec);
  }
  return ret;
}

/*
 * set int list list value to attr.
 */
void AttrValueImpl::SetListListInt(const std::vector<std::vector<int64_t>> &list) {
  auto array = attr_value_->mutable_list_list_int();
  array->clear_list_list_i();
  KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
  for (const std::vector<int64_t> &i : list) {
    const auto list_i = array->add_list_list_i();
    for (const int64_t val : i) {
      list_i->add_list_i(val);
    }
  }
}

/*
 * get float value of attr.
 */
float AttrValueImpl::GetFloat() const { return attr_value_->f(); }

/*
 * get float list value of attr.
 */
std::vector<float> AttrValueImpl::GetListFloat() const {
  std::vector<float> ret;
  auto array = attr_value_->array();
  for (int32_t i = 0; i < array.f_size(); i++) {
    ret.emplace_back(array.f(i));
  }
  return ret;
}

/*
 * attr add float value to list.
 */
void AttrValueImpl::AddListFloat(float f) {
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
  array->add_f(f);
}

/*
 * set float value to attr.
 */
void AttrValueImpl::SetFloat(float f) { attr_value_->set_f(f); }

/*
 * get float list size of attr.
 */
int32_t AttrValueImpl::ListFloatSize() const {
  auto array = attr_value_->array();
  return array.f_size();
}

/*
 * set float list value to attr.
 */
void AttrValueImpl::SetListFloat(const std::vector<float> &list) {
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
  for (const float &f : list) {
    array->add_f(f);
  }
}

/*
 * get bool value of attr.
 */
bool AttrValueImpl::GetBool() const { return attr_value_->b(); }

/*
 * get bool list value of attr.
 */
std::vector<bool> AttrValueImpl::GetListBool() const {
  std::vector<bool> ret;
  auto array = attr_value_->array();
  for (int32_t i = 0; i < array.b_size(); i++) {
    ret.push_back(array.b(i));
  }
  return ret;
}

/*
 * attr add bool value to list.
 */
void AttrValueImpl::AddListBool(bool b) {
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
  array->add_b(b);
}

/*
 * get bool list size of attr.
 */
int32_t AttrValueImpl::ListBoolSize() const {
  auto array = attr_value_->array();
  return array.b_size();
}

/*
 * set bool value to attr.
 */
void AttrValueImpl::SetBool(bool b) { attr_value_->set_b(b); }

/*
 * set bool list value to attr.
 */
void AttrValueImpl::SetListBool(const std::vector<bool> &list) {
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
  for (const bool &b : list) {
    array->add_b(b);
  }
}

/*
 * get data type value of attr.
 */
DataType AttrValueImpl::GetDataType() const { return static_cast<DataType>(attr_value_->type()); }

/*
 * get data type list value of attr.
 */
std::vector<DataType> AttrValueImpl::GetListDataType() const {
  std::vector<DataType> ret;
  auto array = attr_value_->array();
  for (int32_t i = 0; i < array.type_size(); i++) {
    ret.emplace_back(static_cast<DataType>(array.type(i)));
  }
  return ret;
}

/*
 * attr add data type value to list.
 */
void AttrValueImpl::AddListDataType(DataType type) {
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
  array->add_type(type);
}

/*
 * get data type list size of attr.
 */
int32_t AttrValueImpl::ListDataTypeSize() const {
  auto array = attr_value_->array();
  return array.type_size();
}

/*
 * set data type value to attr.
 */
void AttrValueImpl::SetDataType(DataType type) { attr_value_->set_type(type); }

/*
 * set data type list value to attr.
 */
void AttrValueImpl::SetListDataType(const std::vector<DataType> &list) {
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR_VOID(array, "Protobuf mutable array is nullptr")
  for (const DataType &type : list) {
    array->add_type(type);
  }
}

/*
 * set tensor shape value to attr.
 */
bool AttrValueImpl::SetTensorShape(const TensorShape *shape) {
  KERNEL_CHECK_NULLPTR(shape, false, "Shape is null")

  auto tensorShape = attr_value_->mutable_shape();
  KERNEL_CHECK_NULLPTR(tensorShape, false, "Protobuf mutable tensor shape is null")
  auto impl = CpuKernelUtils::GetImpl(shape);
  KERNEL_CHECK_NULLPTR(impl, false, "Get impl is null")
  auto proto = impl->GetProto();
  KERNEL_CHECK_NULLPTR(proto, false, "Get proto is null")
  *tensorShape = *(impl->GetProto());
  return true;
}

/*
 * set tensor shape list value to attr.
 */
uint32_t AttrValueImpl::SetListTensorShape(const std::vector<TensorShape *> &list) {
  uint32_t ret = 0;
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR(array, ret, "Protobuf mutable array is nullptr")

  for (size_t i = 0; i < list.size(); i++) {
    auto tmpShape = array->add_shape();
    if ((list[i] == nullptr) || (tmpShape == nullptr)) {
      KERNEL_LOG_ERROR("Shape[%zu] is null or protobuf add shape ret null.", i);
    } else {
      auto impl = CpuKernelUtils::GetImpl(list[i]);
      if ((impl == nullptr) || (impl->GetProto() == nullptr)) {
        KERNEL_LOG_ERROR("Get list[%zu] impl or proto is null.", i);
        continue;
      }
      *tmpShape = *(impl->GetProto());
      ret++;
    }
  }

  return ret;
}

/*
 * attr add tensor shape value to list.
 */
std::shared_ptr<TensorShape> AttrValueImpl::AddListTensorShape() {
  auto array = attr_value_->mutable_array();
  if (array == nullptr) {
    KERNEL_LOG_ERROR("Protobuf mutable array is nullptr.");
    return std::shared_ptr<TensorShape>(nullptr);
  }

  auto shape = array->add_shape();
  if (shape == nullptr) {
    KERNEL_LOG_ERROR("Protobuf mutable array add shape is nullptr.");
    return std::shared_ptr<TensorShape>(nullptr);
  }

  TensorShapeImpl *impl = new (std::nothrow) TensorShapeImpl(shape);
  if (impl == nullptr) {
    KERNEL_LOG_ERROR("Create TensorShapeImpl failed.");
    return std::shared_ptr<TensorShape>(nullptr);
  }

  auto tensorShape = CpuKernelUtils::CreateTensorShape(impl);
  if (tensorShape == nullptr) {
    delete impl;
  }
  return tensorShape;
}

/*
 * get tensor shape value of attr.
 */
std::shared_ptr<TensorShape> AttrValueImpl::GetTensorShape() const {
  auto shape = attr_value_->mutable_shape();
  if (shape == nullptr) {
    KERNEL_LOG_ERROR("Protobuf mutable shape is nullptr.");
    return std::shared_ptr<TensorShape>(nullptr);
  }

  TensorShapeImpl *impl = new (std::nothrow) TensorShapeImpl(shape);
  if (impl == nullptr) {
    KERNEL_LOG_ERROR("Create TensorShapeImpl failed.");
    return std::shared_ptr<TensorShape>(nullptr);
  }

  auto tensorShape = CpuKernelUtils::CreateTensorShape(impl);
  if (tensorShape == nullptr) {
    delete impl;
  }
  return tensorShape;
}

/*
 * get tensor shape list value of attr.
 */
std::vector<TensorShape> AttrValueImpl::GetListTensorShape() const {
  std::vector<TensorShape> ret;
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR(array, ret, "Protobuf mutable array is nullptr")
  for (int32_t i = 0; i < array->shape_size(); i++) {
    auto shape = array->mutable_shape(i);
    if (shape == nullptr) {
      KERNEL_LOG_ERROR("Protobuf mutable shape[%d] is nullptr.", i);
      return std::vector<TensorShape>();
    }

    TensorShapeImpl *impl = new (std::nothrow) TensorShapeImpl(shape);
    if (impl == nullptr) {
      KERNEL_LOG_ERROR("Create TensorShapeImpl[%d] failed.", i);
      return std::vector<TensorShape>();
    } else {
      auto tensorShape = CpuKernelUtils::CreateTensorShape(impl);
      if (tensorShape == nullptr) {
        delete impl;
        return std::vector<TensorShape>();
      }
      ret.emplace_back(*tensorShape);
    }
  }
  return ret;
}

/*
 * get tensor shape list size of attr.
 */
int32_t AttrValueImpl::ListTensorShapeSize() const {
  auto array = attr_value_->array();
  return array.shape_size();
}

/*
 * set tensor value to attr.
 */
bool AttrValueImpl::SetTensor(const Tensor *tensor) {
  KERNEL_CHECK_NULLPTR(tensor, false, "Tensor is null")
  auto tensorPtr = attr_value_->mutable_tensor();
  KERNEL_CHECK_NULLPTR(tensorPtr, false, "Protobuf mutable tensor is nullptr")
  auto impl = CpuKernelUtils::GetImpl(tensor);
  KERNEL_CHECK_NULLPTR(impl, false, "Get impl is nullptr")
  auto proto = impl->GetProto();
  KERNEL_CHECK_NULLPTR(proto, false, "Get proto is nullptr")
  *tensorPtr = *(proto);
  return true;
}

/*
 * set tensor list value to attr.
 */
uint32_t AttrValueImpl::SetListTensor(const std::vector<Tensor *> &list) {
  uint32_t ret = 0;
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR(array, ret, "Protobuf mutable array is nullptr")
  for (size_t i = 0; i < list.size(); i++) {
    auto tensorPtr = array->add_tensor();
    if ((list[i] == nullptr) || (tensorPtr == nullptr)) {
      KERNEL_LOG_WARN("Tensor[%zu] is null or protobuf add tensor ret null.", i);
    } else {
      auto impl = CpuKernelUtils::GetImpl(list[i]);
      if ((impl == nullptr) || (impl->GetProto() == nullptr)) {
        KERNEL_LOG_WARN("Get list[%zu] impl or proto is null.", i);
        continue;
      }
      *tensorPtr = *(impl->GetProto());
      ret++;
    }
  }
  return ret;
}

/*
 * attr add tensor value to list.
 */
std::shared_ptr<Tensor> AttrValueImpl::AddListTensor() {
  auto array = attr_value_->mutable_array();
  if (array == nullptr) {
    KERNEL_LOG_ERROR("Protobuf mutable array is nullptr.");
    return std::shared_ptr<Tensor>(nullptr);
  }

  auto tensor = array->add_tensor();
  if (tensor == nullptr) {
    KERNEL_LOG_ERROR("Protobuf mutable array add tensor is nullptr.");
    return std::shared_ptr<Tensor>(nullptr);
  }

  TensorImpl *impl = new (std::nothrow) TensorImpl(tensor);
  if (impl == nullptr) {
    KERNEL_LOG_ERROR("Create TensorImpl failed.");
    return std::shared_ptr<Tensor>(nullptr);
  }

  auto aicpuTensor = CpuKernelUtils::CreateTensor(impl);
  if (aicpuTensor == nullptr) {
    delete impl;
  }
  return aicpuTensor;
}

/*
 * get tensor value of attr.
 */
std::shared_ptr<Tensor> AttrValueImpl::GetTensor() const {
  auto tensor = attr_value_->mutable_tensor();
  if (tensor == nullptr) {
    KERNEL_LOG_ERROR("Protobuf mutable tensor is nullptr.");
    return std::shared_ptr<Tensor>(nullptr);
  }

  TensorImpl *impl = new (std::nothrow) TensorImpl(tensor);
  if (impl == nullptr) {
    KERNEL_LOG_ERROR("Create TensorImpl failed.");
    return std::shared_ptr<Tensor>(nullptr);
  }

  auto aicpuTensor = CpuKernelUtils::CreateTensor(impl);
  if (aicpuTensor == nullptr) {
    delete impl;
  }
  return aicpuTensor;
}

/*
 * get tensor list value of attr.
 */
std::vector<Tensor> AttrValueImpl::GetListTensor() const {
  std::vector<Tensor> ret;
  auto array = attr_value_->mutable_array();
  KERNEL_CHECK_NULLPTR(array, ret, "Protobuf mutable array is nullptr")
  for (int32_t i = 0; i < array->tensor_size(); i++) {
    auto tensor = array->mutable_tensor(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR("Protobuf mutable tensor is nullptr.");
      return std::vector<Tensor>();
    }

    TensorImpl *impl = new (std::nothrow) TensorImpl(tensor);
    if (impl == nullptr) {
      KERNEL_LOG_ERROR("Create TensorImpl[%d] failed.", i);
      return std::vector<Tensor>();
    } else {
      auto aicpuTensor = CpuKernelUtils::CreateTensor(impl);
      if (aicpuTensor == nullptr) {
        delete impl;
        return std::vector<Tensor>();
      }
      ret.emplace_back(*aicpuTensor);
    }
  }
  return ret;
}

/*
 * get tensor list size of attr.
 */
int32_t AttrValueImpl::ListTensorSize() const {
  auto array = attr_value_->array();
  return array.tensor_size();
}

/*
 * get attr proto.
 */
aicpuops::AttrValue *AttrValueImpl::GetProto() const { return attr_value_.get(); }
}  // namespace aicpu
