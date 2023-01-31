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
#include "cpu_kernel/cpu_proto/tensor_impl.h"

#include <string>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_kernel/inc/cpu_types.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "proto/cpu_tensor_shape.pb.h"
#include "cpu_kernel/cpu_proto/tensor_shape_impl.h"

namespace aicpu {
/*
 * get tensor shape value of tensor.
 */
std::shared_ptr<TensorShape> TensorImpl::GetTensorShape() const {
  aicpuops::TensorShape *tensor_shape = tensor_->mutable_tensor_shape();
  if (tensor_shape == nullptr) {
    KERNEL_LOG_ERROR("Protobuf mutable tensor shape is null.");
    return std::shared_ptr<TensorShape>(nullptr);
  }

  TensorShapeImpl *impl = new (std::nothrow) TensorShapeImpl(tensor_shape);
  if (impl == nullptr) {
    KERNEL_LOG_ERROR("Create TensorShapeImpl failed.");
    return std::shared_ptr<TensorShape>(nullptr);
  }

  auto aicpu_shape = CpuKernelUtils::CreateTensorShape(impl);
  if (aicpu_shape == nullptr) {
    delete impl;
  }
  return aicpu_shape;
}

/*
 * set tensor shape value to tensor.
 */
bool TensorImpl::SetTensorShape(const TensorShape *shape) {
  KERNEL_CHECK_NULLPTR(shape, false, "Tensor shape is null")

  aicpuops::TensorShape *tensor_shape = tensor_->mutable_tensor_shape();
  KERNEL_CHECK_NULLPTR(tensor_shape, false, "Protobuf mutable tensor shape is null")
  auto impl = CpuKernelUtils::GetImpl(shape);
  KERNEL_CHECK_NULLPTR(impl, false, "Get impl is null")

  auto proto = impl->GetProto();
  KERNEL_CHECK_NULLPTR(proto, false, "Get proto is null")

  *tensor_shape = *(proto);
  return true;
}

/*
 * get data type value of tensor.
 */
DataType TensorImpl::GetDataType() const { return static_cast<DataType>(tensor_->tensor_type()); }

/*
 * set data type value to tensor.
 */
void TensorImpl::SetDataType(DataType type) { tensor_->set_tensor_type(type); }

/*
 * get data ptr of tensor.
 */
void *TensorImpl::GetData() const { return reinterpret_cast<void *>(static_cast<uintptr_t>(tensor_->data_ptr())); }

/*
 * set data ptr to tensor.
 */
void TensorImpl::SetData(void *addr) { tensor_->set_data_ptr(static_cast<uint64_t>(reinterpret_cast<intptr_t>(addr))); }

/*
 * get data size of tensor.
 */
uint64_t TensorImpl::GetDataSize() const { return tensor_->data_size(); }

/*
 * set data size to tensor.
 */
void TensorImpl::SetDataSize(uint64_t size) { tensor_->set_data_size(size); }

/*
 * get name of tensor.
 */
std::string TensorImpl::GetName() const { return tensor_->name(); }

/*
 * set name of tensor.
 */
void TensorImpl::SetName(const std::string &name) { tensor_->set_name(name); }

/*
 * calculate data size by tensor shape.
 */
int64_t TensorImpl::CalcDataSizeByShape() const {
  int64_t data_size = NumElements();
  int32_t element_size = GetSizeByDataType(static_cast<DataType>(GetDataType()));
  if ((data_size < 0) || (element_size < 0)) {
    KERNEL_LOG_WARN("Get tensor element number[%lld] or element type size[%d] less than 0.", data_size, element_size);
    return -1;
  }

  KERNEL_CHECK_ASSIGN_64S_MULTI(data_size, element_size, data_size, -1);
  return data_size;
}

/*
 * get data elements number.
 */
int64_t TensorImpl::NumElements() const {
  auto shape = GetTensorShape();
  if (shape == nullptr) {
    KERNEL_LOG_ERROR("Get tensor shape failed.");
    return -1;
  }

  return shape->NumElements();
}

aicpuops::Tensor *TensorImpl::GetProto() const { return tensor_.get(); }
}  // namespace aicpu