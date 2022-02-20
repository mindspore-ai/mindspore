/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/scatter_nd_max_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kScatterNdMaxInputsNum = 3;
constexpr size_t kScatterNdMaxOutputsNum = 1;
constexpr size_t kNumUnits = 2;
constexpr size_t kInputIndices = 2;
constexpr char kKernelName[] = "ScatterNdMax";

template <typename T, typename S>
void Compute(const ComputeParams<T, S> *params, const size_t start, const size_t end) {
  MS_EXCEPTION_IF_NULL(params);
  T *x = params->x_;
  S *indices = params->indices_;
  T *updates = params->updates_;
  std::vector<int> *out_strides = params->out_strides_;
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(indices);
  MS_EXCEPTION_IF_NULL(updates);
  MS_EXCEPTION_IF_NULL(out_strides);

  for (int i = SizeToInt(start); i < SizeToInt(end); ++i) {
    int offset = 0;
    for (int j = 0; j < params->indices_unit_rank_; ++j) {
      auto index = indices[i * params->indices_unit_rank_ + j];
      offset += index * out_strides->at(j) * params->unit_size_;
    }
    if ((params->indices_unit_rank_ == 1) && (params->unit_size_ != 1)) {
      for (int k = 0; k < static_cast<int>(params->unit_size_); k++) {
        if (x[offset + k] < updates[params->unit_size_ * i + k]) {
          x[offset + k] = updates[params->unit_size_ * i + k];
        }
      }
    } else {
      if (x[offset] < updates[params->unit_size_ * i]) {
        x[offset] = updates[params->unit_size_ * i];
      }
    }
  }
}
}  // namespace

void ScatterMaxCPUKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto indices_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto updates_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
  auto indices_unit_rank = indices_shape.back();
  if (indices_unit_rank > shape.size()) {
    MS_EXCEPTION(ValueError)
      << "For '" << kernel_name_
      << "', the value of last dimension of 'indices' should be less than "
         "or equal to the dimension of 'shape', but got the value of last dimension of 'indices': "
      << indices_unit_rank << " and the dimension of 'shape': " << shape.size();
  }
  if (updates_shape.size() != indices_shape.size() - 1 + shape.size() - indices_unit_rank) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the dimension of 'update' and 'shape', 'indices' are not "
                                "satisfy the equivalence relationship: "
                                "'updates_shape.size() == indices_shape.size() - 1 + shape.size() - indices_unit_rank'";
  }
  indices_unit_rank_ = SizeToInt(indices_unit_rank);
  unit_size_ = 1;
  for (size_t i = indices_shape.size() - 1; i < updates_shape.size(); ++i) {
    unit_size_ *= SizeToInt(updates_shape[i]);
  }
  num_units_ = 1;
  num_units_ *= updates_shape[indices_shape.size() - kNumUnits];
  for (int i = SizeToInt(indices_shape.size()) - 3; i >= 0; i--) {
    num_units_ *= updates_shape[i];
  }
  int out_stride = 1;
  out_strides_.push_back(out_stride);
  for (int i = indices_unit_rank_ - 2; i >= 0; i--) {
    out_stride *= shape[i + 1];
    out_strides_.push_back(out_stride);
  }
  reverse(out_strides_.begin(), out_strides_.end());
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  indices_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
}

bool ScatterMaxCPUKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kScatterNdMaxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kScatterNdMaxOutputsNum, kernel_name_);
  switch (dtype_) {
    case kNumberTypeInt8:
      return DoComputeWithIndicesType<int8_t>(inputs, outputs, indices_dtype_);
    case kNumberTypeInt16:
      return DoComputeWithIndicesType<int16_t>(inputs, outputs, indices_dtype_);
    case kNumberTypeInt32:
      return DoComputeWithIndicesType<int>(inputs, outputs, indices_dtype_);
    case kNumberTypeInt64:
      return DoComputeWithIndicesType<int64_t>(inputs, outputs, indices_dtype_);
    case kNumberTypeUInt8:
      return DoComputeWithIndicesType<uint8_t>(inputs, outputs, indices_dtype_);
    case kNumberTypeUInt16:
      return DoComputeWithIndicesType<uint16_t>(inputs, outputs, indices_dtype_);
    case kNumberTypeUInt32:
      return DoComputeWithIndicesType<uint32_t>(inputs, outputs, indices_dtype_);
    case kNumberTypeUInt64:
      return DoComputeWithIndicesType<uint64_t>(inputs, outputs, indices_dtype_);
    case kNumberTypeFloat16:
      return DoComputeWithIndicesType<float16>(inputs, outputs, indices_dtype_);
    case kNumberTypeFloat32:
      return DoComputeWithIndicesType<float>(inputs, outputs, indices_dtype_);
    case kNumberTypeFloat64:
      return DoComputeWithIndicesType<double>(inputs, outputs, indices_dtype_);
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                              << "', the dtype of 'input_x' should be float16, float32, float64, int8, int16, int32, "
                                 " int64, uint8, uint16, uint32 and uint64, but got "
                              << TypeIdLabel(dtype_);
      return false;
  }
  return true;
}

template <typename T>
bool ScatterMaxCPUKernelMod::DoComputeWithIndicesType(const std::vector<kernel::AddressPtr> &inputs,
                                                      const std::vector<kernel::AddressPtr> &outputs,
                                                      TypeId indices_dtype_) {
  switch (indices_dtype_) {
    case kNumberTypeInt32:
      return LaunchKernel<T, int>(inputs, outputs);
    case kNumberTypeInt64:
      return LaunchKernel<T, int64_t>(inputs, outputs);
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                              << "', the dtype of 'indices' should be int32 and int64, but got "
                              << TypeIdLabel(indices_dtype_);
      return false;
  }
}

template <typename T, typename S>
bool ScatterMaxCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  T *x = reinterpret_cast<T *>(ScatterMaxRealData(inputs, outputs));
  S *indices = reinterpret_cast<S *>(inputs[1]->addr);
  T *updates = reinterpret_cast<T *>(inputs[2]->addr);
  ComputeParams<T, S> params;
  params.x_ = x;
  params.indices_ = indices;
  params.updates_ = updates;
  params.x_mem_size_ = inputs[0]->size / sizeof(T);
  params.unit_size_ = unit_size_;
  params.indices_unit_rank_ = indices_unit_rank_;
  params.out_strides_ = &out_strides_;
  params.indices_mem_size_ = inputs[1]->size / sizeof(S);
  for (int i = 0; i < static_cast<int>(params.indices_mem_size_); i++) {
    if (indices[i] < 0) {
      MS_EXCEPTION(ValueError) << "For '" << kKernelName
                               << "', each element in 'indices' should be greater than or equal to 0, but got "
                               << params.indices_[i];
      return false;
    } else {
      continue;
    }
  }
  std::vector<common::Task> tasks;
  size_t start = 0;
  auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  size_t once_compute_size = 0;
  if (max_thread_num == 0) {
    once_compute_size = num_units_;
  } else {
    once_compute_size = (num_units_ + max_thread_num - 1) / max_thread_num;
  }
  while (start < num_units_) {
    size_t end = (start + once_compute_size) > num_units_ ? num_units_ : (start + once_compute_size);
    auto task = [&params, start, end]() {
      Compute<T, S>(&params, start, end);
      return common::SUCCESS;
    };
    (void)tasks.emplace_back(task);
    start += once_compute_size;
  }
  ParallelLaunch(tasks);
  (void)memcpy_s(outputs[0]->addr, outputs[0]->size, x, inputs[0]->size);
  return true;
}

void *ScatterNdMaxCpuKernelMod::ScatterMaxRealData(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<kernel::AddressPtr> &) {
  return inputs[0]->addr;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterNdMax, ScatterNdMaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
