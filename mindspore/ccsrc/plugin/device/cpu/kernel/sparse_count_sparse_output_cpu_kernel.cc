/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <cstdio>
#include <vector>
#include <map>
#include <limits>
#include <memory>
#include "utils/ms_utils.h"
#include "plugin/device/cpu/kernel/sparse_count_sparse_output_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "ops/sparse_count_sparse_output.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseCountSparseOutputInputsNum = 4;
constexpr size_t kSparseCountSparseOutputOutputsNum = 3;
constexpr size_t kIndicesIndex = 0;
constexpr size_t kValuesIndex = 1;
constexpr size_t kShapeIndex = 2;
constexpr size_t kWeightsIndex = 3;
constexpr size_t kMaxIndicesRank = 2;
}  // namespace

static int kMaxBatches = std::numeric_limits<int>::max();

template <class T>
using BatchedMap = std::vector<std::map<int64_t, T>>;

void SparseCountSparseOutputCpuKernelMod::CheckIndicesInBounds(const int64_t *indices_addr, const int64_t *shape_ptr,
                                                               size_t indices_length, bool is_1d, size_t rank,
                                                               int64_t n_batches) const {
  if (rank == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input rank must be greater than 0, but got 0.";
  }
  indices_length = indices_length / rank;
  for (size_t i = 0; i < rank; i++) {
    if (!is_1d) {
      for (size_t j = 0; j < indices_length; j++) {
        if (indices_addr[i + j * rank] >= shape_ptr[i]) {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input index value " << indices_addr[i + j * rank]
                            << " must be in [0, " << shape_ptr[i] << ") as given by dense shape";
          break;
        }
      }
    } else {
      if (indices_addr[i] >= shape_ptr[i]) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input index value " << indices_addr[i]
                          << " must be in [0, " << shape_ptr[i] << ") as given by dense shape";
        break;
      }
    }
  }

  if (n_batches <= 0 || n_batches > kMaxBatches) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', cannot allocate " << n_batches
                      << " batches, dense shape too wide";
  }
}

template <typename T>
void SparseCountSparseOutputCpuKernelMod::CheckValidValuesAndWeights(const T *values_addr, bool use_weights) const {
  for (size_t i = 0; i < values_size_; ++i) {
    if (values_addr[i] < 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input values must be non-negative";
    }
  }

  if (use_weights) {
    if (weights_shape_[0] != values_shape_[0]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', weights and values must have the same shape";
    }
  }
}

bool SparseCountSparseOutputCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.size() != kSparseCountSparseOutputInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input size must be " << kSparseCountSparseOutputInputsNum
                  << ", but got " << inputs.size();
    return false;
  }
  outputs_ = outputs;
  size_t output_num = outputs.size();
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kSparseCountSparseOutputOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_
                      << "SparseCountSparseOutput does not support this kernel data type: " << kernel_attr
                      << "support kernel input type and format: " << GetOpSupport();
  }
  kernel_func_ = func_list_[index].second;
  auto kernel_ptr = std::make_shared<ops::SparseCountSparseOutput>(base_operator->GetPrim());
  binary_output_ = kernel_ptr->get_binary_output();
  minlength_ = kernel_ptr->get_minlength();
  maxlength_ = kernel_ptr->get_maxlength();

  (void)types_.emplace_back(TypeId::kNumberTypeInt64);
  (void)types_.emplace_back(inputs[1]->GetDtype());
  (void)types_.emplace_back(TypeId::kNumberTypeInt64);
  is_need_retrieve_output_shape_ = true;
  return true;
}

int SparseCountSparseOutputCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  indices_shape_ = inputs[kIndicesIndex]->GetShapeVector();
  values_shape_ = inputs[kValuesIndex]->GetShapeVector();
  shape_shape_ = inputs[kShapeIndex]->GetShapeVector();
  weights_shape_ = inputs[kWeightsIndex]->GetShapeVector();
  values_size_ = LongToSize(values_shape_[0]);
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret == KRET_UNKNOWN_OUT_SHAPE) {
    if (input_size_list_.size() != kSparseCountSparseOutputInputsNum) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', Input size list should be " << kSparseCountSparseOutputInputsNum
                    << ", but got " << input_size_list_.size() << ".";
      return KRET_RESIZE_FAILED;
    }
    output_size_list_.clear();
    auto max_out_size = indices_shape_[0] * indices_shape_[1];
    (void)output_size_list_.emplace_back(max_out_size * kMaxIndicesRank * GetTypeByte(TypeIdToType(types_[kIndex0])));
    (void)output_size_list_.emplace_back(max_out_size * GetTypeByte(TypeIdToType(types_[kIndex1])));
    (void)output_size_list_.emplace_back(kMaxIndicesRank * GetTypeByte(TypeIdToType(types_[kIndex2])));
  }
  return ret;
}

template <typename I, typename T>
bool SparseCountSparseOutputCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> & /* workspace */,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseCountSparseOutputInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseCountSparseOutputOutputsNum, kernel_name_);
  if (outputs[0]->size == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', output memory size must be greater than 0, but got 0.";
    return true;
  }

  const auto *indices_addr = reinterpret_cast<int64_t *>(inputs[0]->addr);
  const auto *values_addr = reinterpret_cast<I *>(inputs[1]->addr);
  const auto *shape_ptr = reinterpret_cast<int64_t *>(inputs[2]->addr);
  const auto *weights = reinterpret_cast<T *>(inputs[3]->addr);
  auto *output_indices = reinterpret_cast<int64_t *>(outputs[0]->addr);
  auto *output_values = reinterpret_cast<T *>(outputs[1]->addr);
  auto *output_shape = reinterpret_cast<int64_t *>(outputs[2]->addr);
  const size_t indices_length = inputs[0]->size / sizeof(int64_t);
  bool use_weights = inputs[3]->size > 0;
  bool is_1d = shape_shape_[0] == 1;
  size_t rank = is_1d ? 1 : indices_shape_[1];

  // Check if values and weights are valid
  CheckValidValuesAndWeights<I>(values_addr, use_weights);

  int64_t n_batches = is_1d ? 1 : shape_ptr[0];
  // Check that index values are in bounds of the dense shape
  CheckIndicesInBounds(indices_addr, shape_ptr, indices_length, is_1d, rank, n_batches);

  int64_t max_val = 0;
  auto per_batch_counts = BatchedMap<T>(shape_ptr[0]);
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      int64_t batch = is_1d ? 0 : indices_addr[i * rank];
      if (batch > n_batches) {
        MS_LOG(EXCEPTION)
          << "For '" << kernel_name_
          << "', Indices value along the first dimension must be lower than the first index of the shape. Got " << batch
          << " as batch and" << n_batches << " as the first dimension of the shape.";
      }
      const auto value = values_addr[i];
      if (maxlength_ < 0 || value < maxlength_) {
        if (binary_output_) {
          per_batch_counts[batch][value] = 1;
        } else if (use_weights) {
          per_batch_counts[batch][value] += weights[i];
        } else {
          per_batch_counts[batch][value]++;
        }
        if (value > max_val) {
          max_val = value;
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, values_size_, this, &parallel_search_info_);

  // Calculate the size of the output
  auto out_size = maxlength_ >= 0 ? maxlength_ : std::max((max_val + 1), minlength_);

  int64_t value_pos = 0;
  auto out_task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      const auto &per_batch_count = per_batch_counts[i];
      std::vector<std::pair<int64_t, T>> ind_val_pairs(per_batch_count.begin(), per_batch_count.end());
      sort(ind_val_pairs.begin(), ind_val_pairs.end());
      for (const auto &x : ind_val_pairs) {
        if (is_1d) {
          output_indices[value_pos] = x.first;
        } else {
          output_indices[value_pos * 2] = i;
          output_indices[value_pos * 2 + 1] = x.first;
        }
        output_values[value_pos] = x.second;
        ++value_pos;
      }
    }
  };
  ParallelLaunchAutoSearch(out_task, per_batch_counts.size(), this, &parallel_search_info_);
  if (is_1d) {
    output_shape[0] = out_size;
  } else {
    output_shape[0] = per_batch_counts.size();
    output_shape[1] = out_size;
  }

  // Update output shape based on number of dimensions
  int64_t num_dim = static_cast<int64_t>(rank) > 1 ? 2 : 1;
  std::vector<int64_t> out_indices_shape = {value_pos, num_dim};
  std::vector<int64_t> out_values_shape = {value_pos};
  std::vector<int64_t> out_dense_shape_shape = {num_dim};
  outputs_[kIndex0]->SetShapeVector(out_indices_shape);
  outputs_[kIndex1]->SetShapeVector(out_values_shape);
  outputs_[kIndex2]->SetShapeVector(out_dense_shape_shape);
  return true;
}

std::vector<std::pair<KernelAttr, SparseCountSparseOutputCpuKernelMod::SparseCountSparseOutputFunc>>
  SparseCountSparseOutputCpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseCountSparseOutputCpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseCountSparseOutputCpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseCountSparseOutputCpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseCountSparseOutputCpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseCountSparseOutputCpuKernelMod::LaunchKernel<int32_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseCountSparseOutputCpuKernelMod::LaunchKernel<int64_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseCountSparseOutputCpuKernelMod::LaunchKernel<int32_t, double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseCountSparseOutputCpuKernelMod::LaunchKernel<int32_t, double>},
};

std::vector<KernelAttr> SparseCountSparseOutputCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)transform(func_list_.begin(), func_list_.end(), back_inserter(support_list),
                  [](const std::pair<KernelAttr, SparseCountSparseOutputFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseCountSparseOutput, SparseCountSparseOutputCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
