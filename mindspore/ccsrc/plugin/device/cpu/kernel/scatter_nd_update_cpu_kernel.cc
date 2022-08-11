/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/scatter_nd_update_cpu_kernel.h"
#include <complex>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kScatterNdUpdateInputsNum = 3;
constexpr size_t kScatterNdUpdateOutputsNum = 1;
constexpr size_t kMinIndiceRank = 2;
constexpr char kKernelName[] = "ScatterNdUpdate";

template <typename T, typename S>
bool Compute(const ComputeParams<T, S> *params, const size_t start, const size_t end) {
  MS_EXCEPTION_IF_NULL(params);
  T *x = params->x_;
  S *indices = params->indices_;
  T *updates = params->updates_;
  std::vector<size_t> *out_strides = params->out_strides_;
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(indices);
  MS_EXCEPTION_IF_NULL(updates);
  MS_EXCEPTION_IF_NULL(out_strides);

  for (size_t i = start; i < end; ++i) {
    size_t offset = 0;
    std::vector<size_t> local_indices;
    for (size_t j = 0; j < params->indices_unit_rank_; ++j) {
      auto index = indices[i * params->indices_unit_rank_ + j];
      (void)local_indices.emplace_back(IntToSize(index));
      if (index < 0) {
        MS_LOG(ERROR) << "For '" << kKernelName
                      << "', each element in 'indices' must be greater than or equal to 0, but got " << index;
        return false;
      }
      offset += IntToSize(index) * out_strides->at(j) * params->unit_size_;
    }
    if (offset * sizeof(T) > params->x_mem_size_) {
      MS_LOG(ERROR) << "For '" << kKernelName
                    << "', indices out of range for input_x. Please check the indices which is " << local_indices;
      return false;
    }
    auto ret = memcpy_s(x + offset, params->x_mem_size_ - offset * sizeof(T), updates + params->unit_size_ * i,
                        params->unit_size_ * sizeof(T));
    if (ret != 0) {
      MS_LOG(ERROR) << "For '" << kKernelName << "', memcpy_s error. Error no: " << ret;
      return false;
    }
  }
  return true;
}

#define COMPUTE_CASE(DTYPE, TYPE1, TYPE2, inputs, outputs) \
  case (DTYPE): {                                          \
    LaunchKernel<TYPE2, TYPE1>(inputs, outputs);           \
    break;                                                 \
  }
}  // namespace

void ScatterUpdateCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto indices_shape_ori = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto updates_shape_ori = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
  if (AnfAlgo::IsShapesDynamic({shape, indices_shape_ori, updates_shape_ori})) {
    return;
  }
  auto indices_shape = Convert2SizeT(indices_shape_ori);
  auto updates_shape = Convert2SizeT(updates_shape_ori);
  auto indices_unit_rank = indices_shape.back();
  if (indices_unit_rank > shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the value of last dimension of 'indices' must be less than "
                         "or equal to the dimension of 'shape', but got the value of last dimension of 'indices': "
                      << indices_unit_rank << " and the dimension of 'shape': " << shape.size();
  }
  if (indices_shape.size() < kMinIndiceRank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'indices' must be at least 2, but got "
                      << indices_shape.size();
  }
  if (updates_shape.size() != indices_shape.size() - 1 + shape.size() - indices_unit_rank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'update' and 'shape', 'indices' are not "
                         "satisfy the equivalence relationship: "
                         "'updates_shape.size() == indices_shape.size() - 1 + shape.size() - indices_unit_rank'";
  }
  for (size_t i = 0; i < indices_shape.size() - 1; ++i) {
    if (updates_shape[i] != indices_shape[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the shape of 'updates' and 'indices' are different in dimension i=" << i
                        << ". The 'updates_shape[i]' is " << updates_shape[i] << " and the 'indices_shape[i]' is "
                        << indices_shape[i];
    }
  }
  indices_unit_rank_ = indices_unit_rank;
  unit_size_ = 1;
  for (size_t i = indices_shape.size() - 1; i < updates_shape.size(); ++i) {
    unit_size_ *= updates_shape[i];
  }
  num_units_ = 1;
  num_units_ *= updates_shape[indices_shape.size() - 2];
  for (int i = SizeToInt(indices_shape.size()) - 3; i >= 0; i--) {
    num_units_ *= updates_shape[i];
  }
  size_t out_stride = 1;
  out_strides_.push_back(out_stride);
  for (int64_t i = SizeToLong(indices_unit_rank_) - 2; i >= 0; i--) {
    out_stride *= LongToSize(shape[LongToSize(i + 1)]);
    out_strides_.push_back(out_stride);
  }
  reverse(out_strides_.begin(), out_strides_.end());
  dtype_value = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  dtype_shape = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
}

template <typename T>
void ScatterUpdateCpuKernelMod::LaunchTypeChoose(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  switch (dtype_value) {
    case kNumberTypeFloat16:
      LaunchKernel<float16, T>(inputs, outputs);
      break;
    case kNumberTypeFloat32:
      LaunchKernel<float, T>(inputs, outputs);
      break;
    case kNumberTypeFloat64:
      LaunchKernel<double, T>(inputs, outputs);
      break;
    case kNumberTypeInt8:
      LaunchKernel<int8_t, T>(inputs, outputs);
      break;
    case kNumberTypeInt16:
      LaunchKernel<int16_t, T>(inputs, outputs);
      break;
    case kNumberTypeInt32:
      LaunchKernel<int, T>(inputs, outputs);
      break;
    case kNumberTypeInt64:
      LaunchKernel<int64_t, T>(inputs, outputs);
      break;
    case kNumberTypeUInt8:
      LaunchKernel<uint8_t, T>(inputs, outputs);
      break;
    case kNumberTypeUInt16:
      LaunchKernel<uint16_t, T>(inputs, outputs);
      break;
    case kNumberTypeUInt32:
      LaunchKernel<uint32_t, T>(inputs, outputs);
      break;
    case kNumberTypeUInt64:
      LaunchKernel<uint64_t, T>(inputs, outputs);
    case kNumberTypeComplex64:
      LaunchKernel<std::complex<float>, T>(inputs, outputs);
      break;
    case kNumberTypeComplex128:
      LaunchKernel<std::complex<double>, T>(inputs, outputs);
      break;
    case kNumberTypeBool:
      LaunchKernel<bool, T>(inputs, outputs);
      break;
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dtype of 'input_x' should be float16, float32, float64, int8, int16, int32, "
                           "int64, uint8, uint16, uint32, uint64, complex64 or complex128.but got "
                        << TypeIdLabel(dtype_value);
  }
}

bool ScatterUpdateCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kScatterNdUpdateInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kScatterNdUpdateOutputsNum, kernel_name_);
  switch (dtype_shape) {
    case kNumberTypeInt32:
      LaunchTypeChoose<int32_t>(inputs, outputs);
      return true;
    case kNumberTypeInt64:
      LaunchTypeChoose<int64_t>(inputs, outputs);
      return true;
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'input_indices' should be int32, int64. "
                        << TypeIdLabel(dtype_shape);
  }
}

template <typename T, typename S>
void ScatterUpdateCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  T *x = reinterpret_cast<T *>(ScatterUpdateRealData(inputs, outputs));
  ComputeParams<T, S> params;
  params.x_ = x;
  params.indices_ = reinterpret_cast<S *>(inputs[1]->addr);
  params.updates_ = reinterpret_cast<T *>(inputs[2]->addr);
  params.x_mem_size_ = inputs[0]->size;
  params.unit_size_ = unit_size_;
  params.indices_unit_rank_ = indices_unit_rank_;
  params.out_strides_ = &out_strides_;

  std::vector<common::Task> tasks;
  size_t start = 0;
  int status = 0;
  if (!Compute<T, S>(&params, start, num_units_)) {
    status = -1;
  }
  if (status == -1) {
    MS_LOG(EXCEPTION) << "Some errors occurred! The error message is as above";
  }
  if (memcpy_s(outputs[0]->addr, outputs[0]->size, x, inputs[0]->size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does memory copy fail.";
  }
}

void *ScatterNdUpdateCpuKernelMod::ScatterUpdateRealData(const std::vector<AddressPtr> &inputs,
                                                         const std::vector<kernel::AddressPtr> &) {
  return inputs[0]->addr;
}

void *TensorScatterUpdateCpuKernelMod::ScatterUpdateRealData(const std::vector<AddressPtr> &inputs,
                                                             const std::vector<kernel::AddressPtr> &outputs) {
  void *x = outputs[0]->addr;
  auto ret = memcpy_s(x, outputs[0]->size, inputs[0]->addr, inputs[0]->size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s error. Error no: " << ret;
  }
  return x;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterNdUpdate, ScatterNdUpdateCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorScatterUpdate, TensorScatterUpdateCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
