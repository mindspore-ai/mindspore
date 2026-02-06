/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/normalize_tuple_index_cpu_kernel.h"
#include <algorithm>
#include <utility>

#include "utils/ms_utils.h"
#include "include/utils/thread_pool.h"
#include "mindspore/ops/infer/normalize_tuple_index.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace normalize_tuple_index_cpu {
bool NormalizeTupleIndexCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  index_types_ = GetValue<string>(primitive_->GetAttr(kAttrOriginIndexType));
  tuple_index_types_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr(kAttrTupleIndexTypes));
  dim_index_ = LongToSize(GetValue<int64_t>(primitive_->GetAttr(kAttrTupleIndexAxis)));
  if (primitive_->HasAttr(kAttrExpandDimsMask)) {
    expand_dims_mask_ = LongToSize(GetValue<int64_t>(primitive_->GetAttr(kAttrExpandDimsMask)));
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

static inline void CheckCopy(void *dest, size_t destMax, const void *src, size_t count, const string &kernel_name) {
  auto cp_ret = memcpy_s(dest, destMax, src, count);
  if (cp_ret != EOK) {
    MS_LOG(EXCEPTION) << "For " << kernel_name << ", memcpy error, errorno: " << cp_ret;
  }
}

template <typename T>
void NormalizeTupleIndexCpuKernelMod::NormalizeIntIndex(const ShapeVector &data_shape, int64_t *output_addr,
                                                        const T *index_val_addr, size_t dim_index) {
  int64_t index_val = index_val_addr[0];
  auto new_dim_index =
    ops::NormalizeTupleIndex::NormalizeDimIndex(data_shape, dim_index, tuple_index_types_, expand_dims_mask_);
  auto dim = data_shape[new_dim_index];
  auto new_index_val = ops::NormalizeTupleIndex::CheckRange(index_val, dim);
  CheckCopy(output_addr, sizeof(int64_t), &new_index_val, sizeof(int64_t), kernel_name_);
}

template <typename T>
void NormalizeTupleIndexCpuKernelMod::NormalizeSequenceIndex(const ShapeVector &data_shape, int64_t *output_addr,
                                                             const T *index_val_addr, size_t seq_size,
                                                             size_t dim_index) {
  auto new_dim_index =
    ops::NormalizeTupleIndex::NormalizeDimIndex(data_shape, dim_index, tuple_index_types_, expand_dims_mask_);
  auto dim = data_shape[new_dim_index];
  std::vector<int64_t> out;
  for (size_t i = 0; i < seq_size; i++) {
    int64_t int_index_val = index_val_addr[i];
    int_index_val = ops::NormalizeTupleIndex::CheckRange(int_index_val, dim);
    (void)out.emplace_back(int_index_val);
  }

  const auto output_size = seq_size * sizeof(int64_t);
  CheckCopy(output_addr, output_size, out.data(), output_size, kernel_name_);
  (void)output_sizes_.emplace_back(out.size());
}

template <typename T>
void NormalizeTupleIndexCpuKernelMod::NormalizeBoolSequenceIndex(const ShapeVector &data_shape, int64_t *output_addr,
                                                                 const T *index_val_addr, size_t seq_size,
                                                                 size_t dim_index) {
  auto new_dim_index =
    ops::NormalizeTupleIndex::NormalizeDimIndex(data_shape, dim_index, tuple_index_types_, expand_dims_mask_);
  std::vector<int64_t> out;
  int64_t dim_size = data_shape[new_dim_index];
  if (SizeToLong(seq_size) != dim_size) {
    MS_EXCEPTION(IndexError) << "dimension is " << dim_size << " but corresponding boolean dimension is " << seq_size;
  }
  for (size_t i = 0; i < seq_size; i++) {
    if (index_val_addr[i]) {
      (void)out.emplace_back(SizeToLong(i));
    }
  }
  if (out.empty()) {
    MS_EXCEPTION(IndexError) << "The sequence element(tuple/list) in tuple index can't be empty.";
  }
  const auto output_size = out.size() * sizeof(int64_t);
  CheckCopy(output_addr, output_size, out.data(), output_size, kernel_name_);
  (void)output_sizes_.emplace_back(out.size());
}

int NormalizeTupleIndexCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  (void)KernelMod::Resize(inputs, outputs);
  data_shapes_ = GetShapes(inputs);
  return KRET_OK;
}

void NormalizeTupleIndexCpuKernelMod::NormalizeNoneIndex(int64_t *output_addr, const ShapeVector &data_shape,
                                                         size_t dim_index) {
  auto new_dim_index = ops::NormalizeTupleIndex::NormalizeDimIndex(data_shape, dim_index, tuple_index_types_, 0);
  int64_t dim_size = data_shape[new_dim_index];
  std::vector<int64_t> out;
  for (int64_t i = 0; i < dim_size; i++) {
    (void)out.emplace_back(i);
  }
  size_t output_size = sizeof(int64_t) * out.size();
  CheckCopy(output_addr, output_size, out.data(), output_size, kernel_name_);
  (void)output_sizes_.emplace_back(out.size());
}

void NormalizeTupleIndexCpuKernelMod::NormalizeEllipsisIndex(int64_t *output_addr, const ShapeVector &data_shape,
                                                             size_t dim_index) {
  size_t ellipse_position = 0;
  size_t not_ellipse_occupy_dims = 0;
  for (size_t i = 0; i < kMaxTensorIndexDimNums; i++) {
    if (tuple_index_types_[i] == kMetaTypeEllipsis) {
      ellipse_position = i;
    } else if (tuple_index_types_[i] != kTypeUnknown) {
      not_ellipse_occupy_dims += 1;
    }
  }
  size_t ellipse_occupy_dims = data_shape.size() - not_ellipse_occupy_dims;
  std::vector<int64_t> out;
  if (dim_index >= ellipse_occupy_dims) {
    (void)out.emplace_back(1);
    size_t output_size = sizeof(int64_t) * out.size();
    CheckCopy(output_addr, output_size, out.data(), output_size, kernel_name_);
    (void)output_sizes_.emplace_back(out.size());
    return;
  }
  size_t ellipse_occupy_dims_i = ellipse_position + dim_index;
  int64_t ellipse_occupy_dim = data_shape[ellipse_occupy_dims_i];
  for (int64_t i = 0; i < ellipse_occupy_dim; i++) {
    (void)out.emplace_back(i);
  }
  size_t output_size = sizeof(int64_t) * out.size();
  CheckCopy(output_addr, output_size, out.data(), output_size, kernel_name_);
  (void)output_sizes_.emplace_back(out.size());
}

template <typename T>
bool NormalizeTupleIndexCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &workspace,
                                                   const std::vector<KernelTensor *> &outputs) {
  const auto index_val_addr = GetDeviceAddress<T>(inputs, kIndex1);
  auto output_addr = GetDeviceAddress<int64_t>(outputs, kIndex0);
  const ShapeVector &data_shape = data_shapes_[0];
  output_sizes_.clear();
  if (index_types_ == kIntIndex) {
    NormalizeIntIndex(data_shape, output_addr, index_val_addr, dim_index_);
  } else if (index_types_ == kTensorIndexSequenceIndex) {
    size_t seq_size = static_cast<size_t>(data_shapes_[kIndex1][0]);
    NormalizeSequenceIndex(data_shape, output_addr, index_val_addr, seq_size, dim_index_);
  } else if (index_types_ == kBoolSequenceIndex) {
    size_t seq_size = static_cast<size_t>(data_shapes_[kIndex1][0]);
    NormalizeBoolSequenceIndex(data_shape, output_addr, index_val_addr, seq_size, dim_index_);
  } else if (index_types_ == kNoneIndex) {
    NormalizeNoneIndex(output_addr, data_shape, dim_index_);
  } else if (index_types_ == kEllipsisIndex) {
    NormalizeEllipsisIndex(output_addr, data_shape, dim_index_);
  }
  return true;
}

bool NormalizeTupleIndexCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs) {
  return kernel_func_(this, inputs, workspace, outputs);
}

std::vector<std::pair<KernelAttr, NormalizeTupleIndexCpuKernelMod::NormalizeTupleIndexFunc>>
  NormalizeTupleIndexCpuKernelMod::func_list_ = {};

std::vector<KernelAttr> NormalizeTupleIndexCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    ops::all_type_ids.begin(), ops::all_type_ids.end(), std::back_inserter(func_list_),
    [](TypeId data_type_id) -> std::pair<KernelAttr, NormalizeTupleIndexFunc> {
      return {KernelAttr().AddInputAttr(data_type_id).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
              &NormalizeTupleIndexCpuKernelMod::LaunchKernel<int64_t>};
    });
  (void)std::transform(
    ops::all_type_ids.begin(), ops::all_type_ids.end(), std::back_inserter(func_list_),
    [](TypeId data_type_id) -> std::pair<KernelAttr, NormalizeTupleIndexFunc> {
      return {KernelAttr().AddInputAttr(data_type_id).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64),
              &NormalizeTupleIndexCpuKernelMod::LaunchKernel<bool>};
    });
  (void)std::transform(ops::all_type_ids.begin(), ops::all_type_ids.end(), std::back_inserter(func_list_),
                       [](TypeId data_type_id) -> std::pair<KernelAttr, NormalizeTupleIndexFunc> {
                         return {KernelAttr()
                                   .AddInputAttr(data_type_id)
                                   .AddInputAttr(kObjectTypeTuple, kNumberTypeBool)
                                   .AddOutputAttr(kNumberTypeInt64),
                                 &NormalizeTupleIndexCpuKernelMod::LaunchKernel<bool>};
                       });
  std::transform(ops::all_type_ids.begin(), ops::all_type_ids.end(), std::back_inserter(func_list_),
                 [](TypeId data_type_id) -> std::pair<KernelAttr, NormalizeTupleIndexFunc> {
                   return {KernelAttr()
                             .AddInputAttr(data_type_id)
                             .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                             .AddOutputAttr(kNumberTypeInt64),
                           &NormalizeTupleIndexCpuKernelMod::LaunchKernel<bool>};
                 });
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NormalizeTupleIndexFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NormalizeTupleIndex, NormalizeTupleIndexCpuKernelMod);
}  // namespace normalize_tuple_index_cpu
}  // namespace kernel
}  // namespace mindspore
