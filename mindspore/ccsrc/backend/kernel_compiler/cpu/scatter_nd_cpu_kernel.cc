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

#include "backend/kernel_compiler/cpu/scatter_nd_cpu_kernel.h"
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kScatterNdInputSize = 2;
constexpr size_t kScatterNdOutputSize = 1;
constexpr size_t kMinIndiceRank = 2;

template <typename S, typename T>
void Compute(const ComputeParams<S, T> *params, const size_t start, const size_t end) {
  T *target = params->target_;
  S *indices = params->indices_;
  T *updates = params->updates_;
  std::vector<int> *out_strides = params->out_strides_;

  for (size_t i = start; i < end; ++i) {
    int offset = 0;
    for (size_t j = 0; j < IntToSize(params->indices_unit_rank_); ++j) {
      int index = static_cast<int>(indices[i * IntToSize(params->indices_unit_rank_) + j]);
      if (index < 0) {
        MS_LOG(EXCEPTION) << "Indices contains element " << index << " less than 0.";
      }
      offset += index * out_strides->at(j) * params->unit_size_;
    }
    auto task = [&target, &updates, &params, offset, i](size_t update_start, size_t update_end) {
      for (size_t idx = update_start; idx < update_end; idx++) {
        target[IntToSize(offset) + idx] += updates[IntToSize(params->unit_size_) * i + idx];
      }
    };
    CPUKernelUtils::ParallelFor(task, IntToSize(params->unit_size_));
  }
}
}  // namespace

template <typename S, typename T>
void ScatterNdCPUKernel<S, T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  Check(kernel_node);
  auto shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto updates_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto indices_unit_rank = indices_shape.back();
  if (indices_unit_rank > shape.size()) {
    MS_LOG(EXCEPTION) << "Value of last dimension of indices is greater than shape rank";
  }
  if (indices_shape.size() < kMinIndiceRank) {
    MS_LOG(EXCEPTION) << "Indices has dimension less than 2";
  }
  if (updates_shape.size() != indices_shape.size() - 1 + shape.size() - indices_unit_rank) {
    MS_LOG(EXCEPTION) << "The ranks of update and indices are inconsistent";
  }
  for (size_t i = 0; i < indices_shape.size() - 1; ++i) {
    if (updates_shape[i] != indices_shape[i]) {
      MS_LOG(EXCEPTION) << "The shape of updates and indices are different in dimension " << i << " .";
    }
  }
  indices_unit_rank_ = SizeToInt(indices_unit_rank);
  for (size_t i = indices_shape.size() - 1; i < updates_shape.size(); ++i) {
    unit_size_ *= SizeToInt(updates_shape[i]);
  }

  num_units_ *= updates_shape[indices_shape.size() - 2];
  for (int i = SizeToInt(indices_shape.size()) - 3; i >= 0; i--) {
    num_units_ *= updates_shape[IntToSize(i)];
  }
  int out_stride = 1;
  out_strides_.push_back(out_stride);
  for (int i = indices_unit_rank_ - 2; i >= 0; i--) {
    out_stride *= SizeToInt(shape[IntToSize(i + 1)]);
    out_strides_.push_back(out_stride);
  }
  reverse(out_strides_.begin(), out_strides_.end());
}

template <typename S, typename T>
bool ScatterNdCPUKernel<S, T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  auto target = reinterpret_cast<T *>(outputs[0]->addr);
  auto target_init = memset_s(target, outputs[0]->size, 0, outputs[0]->size);
  if (target_init != EOK) {
    MS_LOG(EXCEPTION) << "ScatterNdCPUKernel Launch task memset failed.";
    return false;
  }
  ComputeParams<S, T> params;
  params.target_ = target;
  params.indices_ = reinterpret_cast<S *>(inputs[0]->addr);
  params.updates_ = reinterpret_cast<T *>(inputs[1]->addr);
  params.target_mem_size_ = outputs[0]->size;
  params.unit_size_ = unit_size_;
  params.indices_unit_rank_ = indices_unit_rank_;
  params.out_strides_ = &out_strides_;

  auto task = [this, &params](size_t start, size_t end) {
    for (size_t idx = start; idx < end; idx++) {
      Compute<S, T>(&params, idx, idx + 1);
    }
  };
  CPUKernelUtils::ParallelFor(task, num_units_);
  return true;
}

template <typename S, typename T>
void ScatterNdCPUKernel<S, T>::Check(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != kScatterNdInputSize) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but ScatterNd needs 2 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != kScatterNdOutputSize) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but ScatterNd needs 1 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
