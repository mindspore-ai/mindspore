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

#include "backend/kernel_compiler/cpu/transpose_cpu_kernel.h"
#include <vector>
#include "runtime/device/cpu/cpu_device_address.h"
#include "common/thread_pool.h"
#include "nnacl/fp32/transpose_fp32.h"
#include "nnacl/int8/transpose_int8.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTransposeInputsNum = 1;
constexpr size_t kTransposeOutputsNum = 1;

// kMaxTransposeSerialSize = 64 * 3 * 512 * 512
constexpr size_t kMaxTransposeSerialSize = 50331648;
}  // namespace

void TransposeCPUFwdKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  auto tmp = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "perm");
  axes_ = {tmp.begin(), tmp.end()};
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (axes_.size() > MAX_TRANSPOSE_DIM_SIZE) {
    MS_LOG(EXCEPTION) << "Transpose support max dimension is " << MAX_TRANSPOSE_DIM_SIZE << "D, but got "
                      << axes_.size() << "D.";
  }

  for (size_t i = 0; i < axes_.size(); ++i) {
    transpose_param_.perm_[i] = SizeToInt(axes_[i]);
  }
  int num_axes = SizeToInt(input_shape_.size());
  transpose_param_.perm_size_ = axes_.size();
  transpose_param_.num_axes_ = num_axes;
  transpose_param_.strides_[num_axes - 1] = 1;
  transpose_param_.out_strides_[num_axes - 1] = 1;
  for (int i = num_axes - 2; i >= 0; i--) {
    transpose_param_.strides_[i] = SizeToInt(input_shape_[i + 1]) * transpose_param_.strides_[i + 1];
    transpose_param_.out_strides_[i] = SizeToInt(output_shape_[i + 1]) * transpose_param_.out_strides_[i + 1];
  }
  launch_map_[kNumberTypeInt8] = &TransposeCPUFwdKernel::LaunchKernel<int8_t>;
  launch_map_[kNumberTypeInt16] = &TransposeCPUFwdKernel::LaunchKernel<int16_t>;
  launch_map_[kNumberTypeInt32] = &TransposeCPUFwdKernel::LaunchKernel<int>;
  launch_map_[kNumberTypeInt64] = &TransposeCPUFwdKernel::LaunchKernel<int64_t>;
  launch_map_[kNumberTypeUInt8] = &TransposeCPUFwdKernel::LaunchKernel<uint8_t>;
  launch_map_[kNumberTypeUInt16] = &TransposeCPUFwdKernel::LaunchKernel<uint16_t>;
  launch_map_[kNumberTypeUInt32] = &TransposeCPUFwdKernel::LaunchKernel<uint32_t>;
  launch_map_[kNumberTypeUInt64] = &TransposeCPUFwdKernel::LaunchKernel<uint64_t>;
  launch_map_[kNumberTypeFloat32] = &TransposeCPUFwdKernel::LaunchKernel<float>;
  launch_map_[kNumberTypeBool] = &TransposeCPUFwdKernel::LaunchKernel<bool>;

  auto iter = launch_map_.find(dtype_);
  if (iter != launch_map_.end()) {
    launch_func_ = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "Unsupported input data type: " << dtype_;
  }
}

bool TransposeCPUFwdKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  launch_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void TransposeCPUFwdKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  const auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  transpose_param_.data_num_ = SizeToInt(inputs[0]->size / sizeof(T));
  int output_shape[SizeToInt(output_shape_.size())];
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    output_shape[i] = SizeToInt(output_shape_[i]);
  }
  size_t data_count = (inputs[0]->size) / sizeof(T);
  if (axes_.size() > DIMENSION_6D || data_count >= kMaxTransposeSerialSize) {
    ParallelRun(input_addr, output_addr, output_shape, data_count);
    return;
  }
  int res = static_cast<int>(NNACL_OK);
  if constexpr (std::is_same_v<T, int8_t>) {
    res = DoTransposeInt8(input_addr, output_addr, output_shape, &transpose_param_);
  } else if constexpr (std::is_same_v<T, int16_t>) {
    res = DoTransposeInt16(input_addr, output_addr, output_shape, &transpose_param_);
  } else if constexpr (std::is_same_v<T, int32_t>) {
    res = DoTransposeInt32(input_addr, output_addr, output_shape, &transpose_param_);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    res = DoTransposeInt64(input_addr, output_addr, output_shape, &transpose_param_);
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    res = DoTransposeUInt8(input_addr, output_addr, output_shape, &transpose_param_);
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    res = DoTransposeUInt16(input_addr, output_addr, output_shape, &transpose_param_);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    res = DoTransposeUInt32(input_addr, output_addr, output_shape, &transpose_param_);
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    res = DoTransposeUInt64(input_addr, output_addr, output_shape, &transpose_param_);
  } else if constexpr (std::is_same_v<T, float>) {
    res = DoTransposeFp32(input_addr, output_addr, output_shape, &transpose_param_);
  } else if constexpr (std::is_same_v<T, bool>) {
    res = DoTransposeBool(input_addr, output_addr, output_shape, &transpose_param_);
  } else {
    MS_LOG(EXCEPTION) << "Transpose not support the type: " << typeid(T).name();
  }
  if (res != static_cast<int>(NNACL_OK)) {
    MS_LOG(EXCEPTION) << "Transpose run failed.";
  }
}

template <typename T>
void TransposeCPUFwdKernel::ParallelRun(const T *input_addr, T *output_addr, const int *output_shape, size_t count) {
  auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  const float block_size = 128.0;
  const size_t thread_num =
    count < block_size * max_thread_num ? FloatToSize(std::ceil(count / block_size)) : max_thread_num;
  std::vector<common::Task> tasks;
  std::function<void(const T *, T *, const int *, TransposeParameter *, int, int)> TransposeDims;

  if constexpr (std::is_same_v<T, int8_t>) {
    TransposeDims = &TransposeDimsInt8;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    TransposeDims = &TransposeDimsInt16;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    TransposeDims = &TransposeDimsInt32;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    TransposeDims = &TransposeDimsInt64;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    TransposeDims = &TransposeDimsUInt8;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    TransposeDims = &TransposeDimsUInt16;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    TransposeDims = &TransposeDimsUInt32;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    TransposeDims = &TransposeDimsUInt64;
  } else if constexpr (std::is_same_v<T, float>) {
    TransposeDims = &TransposeDimsFp32;
  } else if constexpr (std::is_same_v<T, bool>) {
    TransposeDims = &TransposeDimsBool;
  }
  for (int task_id = 0; task_id < SizeToInt(thread_num); ++task_id) {
    auto task = [this, &TransposeDims, &input_addr, &output_addr, &output_shape, task_id, thread_num]() {
      TransposeDims(input_addr, output_addr, output_shape, &transpose_param_, task_id, SizeToInt(thread_num));
      return common::SUCCESS;
    };
    (void)tasks.emplace_back(task);
  }
  (void)common::ThreadPool::GetInstance().SyncRun(tasks);
}
}  // namespace kernel
}  // namespace mindspore
