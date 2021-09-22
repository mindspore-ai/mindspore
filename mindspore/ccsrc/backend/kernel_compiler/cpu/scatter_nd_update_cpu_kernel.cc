/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/scatter_nd_update_cpu_kernel.h"
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kScatterNdUpdateInputsNum = 3;
constexpr size_t kScatterNdUpdateOutputsNum = 1;
constexpr size_t kMinIndiceRank = 2;

template <typename T>
void Compute(const ComputeParams<T> *params, const size_t start, const size_t end) {
  MS_EXCEPTION_IF_NULL(params);
  T *x = params->x_;
  int *indices = params->indices_;
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
      if (index < 0) {
        MS_LOG(EXCEPTION) << "Error, Indices exist element which less than 0. element=" << index;
      }
      offset += index * out_strides->at(j) * params->unit_size_;
    }
    auto ret = memcpy_s(x + offset, params->x_mem_size_ - offset, updates + params->unit_size_ * i,
                        params->unit_size_ * sizeof(T));
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno" << ret;
    }
  }
}
}  // namespace

void ScatterNdUpdateCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto updates_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
  auto indices_unit_rank = indices_shape.back();
  if (indices_unit_rank > shape.size()) {
    MS_LOG(EXCEPTION) << "Value of last dimension of indices is greater than shape rank";
  }
  if (indices_shape.size() < kMinIndiceRank) {
    MS_LOG(EXCEPTION) << "Indices dimension less than 2";
  }
  if (updates_shape.size() != indices_shape.size() - 1 + shape.size() - indices_unit_rank) {
    MS_LOG(EXCEPTION) << "Update, shape rank and indices rank inconsistent";
  }
  for (size_t i = 0; i < indices_shape.size() - 1; ++i) {
    if (updates_shape[i] != indices_shape[i]) {
      MS_LOG(EXCEPTION) << "Value of " << i << "th dimension of indices is not equal to that update";
    }
  }
  indices_unit_rank_ = SizeToInt(indices_unit_rank);
  unit_size_ = 1;
  for (size_t i = indices_shape.size() - 1; i < updates_shape.size(); ++i) {
    unit_size_ *= SizeToInt(updates_shape[i]);
  }
  num_units_ = 1;
  num_units_ *= updates_shape[indices_shape.size() - 2];
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
}

bool ScatterNdUpdateCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kScatterNdUpdateInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kScatterNdUpdateOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported input data type: " << dtype_;
  }
  return true;
}

template <typename T>
void ScatterNdUpdateCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  auto x = reinterpret_cast<T *>(inputs[0]->addr);
  ComputeParams<T> params;
  params.x_ = x;
  params.indices_ = reinterpret_cast<int *>(inputs[1]->addr);
  params.updates_ = reinterpret_cast<T *>(inputs[2]->addr);
  params.x_mem_size_ = inputs[0]->size;
  params.unit_size_ = unit_size_;
  params.indices_unit_rank_ = indices_unit_rank_;
  params.out_strides_ = &out_strides_;

  std::vector<common::Task> tasks;
  size_t start = 0;
  auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  size_t once_compute_size = (num_units_ + max_thread_num - 1) / max_thread_num;
  while (start < num_units_) {
    size_t end = (start + once_compute_size) > num_units_ ? num_units_ : (start + once_compute_size);
    auto task = [&params, start, end]() -> int {
      Compute<T>(&params, start, end);
      return common::SUCCESS;
    };
    (void)tasks.emplace_back(task);
    start += once_compute_size;
  }
  (void)common::ThreadPool::GetInstance().SyncRun(tasks);

  auto ret = memcpy_s(outputs[0]->addr, outputs[0]->size, x, inputs[0]->size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno" << ret;
  }
}
}  // namespace kernel
}  // namespace mindspore
