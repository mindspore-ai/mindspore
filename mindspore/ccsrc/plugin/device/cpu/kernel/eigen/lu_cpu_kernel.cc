/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/eigen/lu_cpu_kernel.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <tuple>
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLUInputsNum = 1;
constexpr size_t kLUaIndex = 0;
constexpr size_t kLUOutputsNum = 3;
constexpr size_t kLuIndex = 0;
constexpr size_t kPivotsIndex = 1;
constexpr size_t kPermutationIndex = 2;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
constexpr int kZeroThreshold = INT32_MIN;
}  // namespace

void LUCpuKernelMod::InitMatrixInfo(const std::vector<size_t> &shape, size_t *row, size_t *col) {
  constexpr size_t lu_min_dim = 1;
  if (shape.size() <= lu_min_dim) {
    MS_LOG_EXCEPTION << kernel_name_ << "shape is " << shape.size() << " which is invalid.";
  }
  constexpr size_t lu_reverse_row_dim = 2;
  *row = shape.at(shape.size() - lu_reverse_row_dim);
  *col = shape.at(shape.size() - 1);
  batch_size_ = lu_min_dim;
  for (int batch = 0; batch < static_cast<int>(shape.size() - lu_reverse_row_dim); ++batch) {
    batch_size_ *= shape.at(SizeToInt(batch));
  }
}

void LUCpuKernelMod::InitPivotVecInfo(const std::vector<size_t> &shape, size_t *row, size_t *col) const {
  constexpr size_t pivot_min_dim = 1;
  if (shape.size() < pivot_min_dim) {
    MS_LOG_EXCEPTION << kernel_name_ << "pivots shape is " << shape.size() << " which is invalid.";
  }
  *row = 1;
  if (shape.size() == pivot_min_dim) {
    *col = shape.front();
  } else {
    *col = shape.back();
  }
}

bool LUCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                          const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  dtype_ = inputs[0]->GetDtype();
  size_t input_num = inputs.size();
  CHECK_KERNEL_INPUTS_NUM(input_num, kLUInputsNum, kernel_name_);
  size_t output_num = outputs.size();
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kLUOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int LUCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                           const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  a_row_ = 1;
  a_col_ = 1;
  lu_row_ = 1;
  lu_col_ = 1;
  permutation_row_ = 1;
  permutation_col_ = 1;
  pivots_row_ = 1;
  pivots_col_ = 1;
  auto a_shape = Convert2SizeTClipNeg(inputs[kLUaIndex]->GetShapeVector());
  InitMatrixInfo(a_shape, &a_row_, &a_col_);
  auto lu_shape = Convert2SizeTClipNeg(outputs[kLuIndex]->GetShapeVector());
  InitMatrixInfo(lu_shape, &lu_row_, &lu_col_);
  auto permutation_shape = Convert2SizeTClipNeg(outputs[kPermutationIndex]->GetShapeVector());
  InitMatrixInfo(permutation_shape, &permutation_row_, &permutation_col_);
  auto pivots_shape = Convert2SizeTClipNeg(outputs[kPivotsIndex]->GetShapeVector());
  InitPivotVecInfo(pivots_shape, &pivots_row_, &pivots_col_);

  size_t lu_size = lu_col_ * dtype_;
  (void)workspace_size_list_.emplace_back(lu_size);
  (void)workspace_size_list_.emplace_back(lu_size);
  return KRET_OK;
}

template <typename T>
T LUCpuKernelMod::GetPermutatedValue(const T *lu_value, const std::vector<int> &per_value, size_t i, size_t j) const {
  const T *pered_lu_value = lu_value + per_value[i] * SizeToInt(lu_col_) + SizeToInt(j);
  return *pered_lu_value;
}

template <typename T>
bool LUCpuKernelMod::UpdateMajorPermutation(T *lu_value, std::vector<int> *per_value, int *pivots, size_t k,
                                            size_t rows) const {
  T max_major_value = static_cast<T>(kZeroThreshold);
  size_t max_major_index = 0;
  for (size_t i = k; i < rows; ++i) {
    T value = GetPermutatedValue(lu_value, *per_value, i, k);
    T abs_value = std::abs(value);
    if (abs_value > max_major_value) {
      max_major_value = abs_value;
      max_major_index = i;
    }
  }
  int per_k = per_value->at(k);
  (*per_value)[k] = per_value->at(max_major_index);
  (*per_value)[max_major_index] = per_k;
  pivots[k] = SizeToInt(max_major_index);
  return max_major_value != static_cast<T>(kZeroThreshold);
}

void LUCpuKernelMod::DoSafeMemCopy(void *dest, size_t dest_max, const void *src, size_t count) const {
  if (memcpy_s(dest, dest_max, src, count) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' It does memory copy failed.";
  }
}

template <typename T>
void LUCpuKernelMod::SetPermutatedValue(T *lu_value, const std::vector<int> &per_value, size_t i, size_t j,
                                        const T &value) const {
  T *per_lu_value = lu_value + per_value[i] * SizeToInt(lu_col_) + SizeToInt(j);
  *per_lu_value = value;
}

template <typename T>
bool LUCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &workspace,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  // input matrix of (m,n) PA = LU
  T *batch_a_value = reinterpret_cast<T *>(inputs[kLUaIndex]->addr);
  T *batch_lu_value = reinterpret_cast<T *>(outputs[kLuIndex]->addr);
  batch_pivots_ = reinterpret_cast<int *>(outputs[kPivotsIndex]->addr);
  int *batch_permutation_value = reinterpret_cast<int *>(outputs[kPermutationIndex]->addr);
  T *lu_ori_wk = reinterpret_cast<T *>(workspace[kLuIndex]->addr);
  T *lu_trans_wk = reinterpret_cast<T *>(workspace[kPivotsIndex]->addr);
  for (size_t batch = 0; batch < batch_size_; ++batch) {
    T *a_value = batch_a_value + batch * a_row_ * a_col_;
    T *lu_value = batch_lu_value + batch * lu_row_ * lu_col_;
    // pivots permutation value
    int *pivots = batch_pivots_ + batch * pivots_row_ * pivots_col_;
    // permutation matrix value
    int *permutation_value = batch_permutation_value + batch * permutation_row_ * permutation_col_;
    // init pivots
    std::vector<int> per_value(pivots_col_, 0);
    for (size_t i = 0; i < pivots_col_; ++i) {
      per_value[i] = SizeToInt(i);
    }
    // 1. memcpy input to output, do full lu inplace.
    DoSafeMemCopy(lu_value, lu_row_ * lu_col_ * sizeof(T), a_value, a_row_ * a_col_ * sizeof(T));
    size_t s = std::min(a_row_, a_col_);
    // 2. do lu decompose inplace
    for (size_t k = 0; k < s; ++k) {
      // 2.1 choose major element of current col if return false means current col elements are all zero, just continue.
      if (!UpdateMajorPermutation(lu_value, &per_value, pivots, k, lu_row_)) {
        continue;
      }
      // 2.2 major element x --> (1/x), get inplace origin lu matrix value.
      T value = static_cast<T>(1.0 / GetPermutatedValue(lu_value, per_value, k, k));
      // 2.3 change major col values
      for (size_t i = k + 1; i < lu_row_; ++i) {
        T y = static_cast<T>(GetPermutatedValue(lu_value, per_value, i, k) * value);
        // set inplace new lu matrix value.
        SetPermutatedValue(lu_value, per_value, i, k, y);
      }

      // 2.4 Gauss elimination core
      for (size_t i = k + 1; i < lu_row_; ++i) {
        for (size_t j = k + 1; j < lu_col_; ++j) {
          T y = static_cast<T>(GetPermutatedValue(lu_value, per_value, i, j) -
                               GetPermutatedValue(lu_value, per_value, i, k) *
                                 GetPermutatedValue(lu_value, per_value, k, j));
          SetPermutatedValue(lu_value, per_value, i, j, y);
        }
      }
    }

    // 3. calculate final lu by permutation list
    std::unordered_map<int, std::pair<int, bool>> pivots_map;
    for (size_t i = 0; i < lu_row_; ++i) {
      pivots_map[per_value[i]] = {SizeToInt(i), false};
    }
    int pivots_count = 0;
    for (const auto &pivot : pivots_map) {
      pivots_count++;
      int key = pivot.first;
      int index = pivot.second.first;
      bool is_visited = pivot.second.second;
      if (is_visited || index == (pivots_count - 1)) {
        continue;
      }

      T *lu_ori_row = lu_value + index * SizeToInt(lu_col_);
      T *lu_trans_row = lu_value + key * SizeToInt(lu_col_);
      // copy ori data to trans lu
      DoSafeMemCopy(lu_trans_wk, lu_col_ * sizeof(T), lu_ori_row, lu_col_ * sizeof(T));
      // copy new data to ori data ptr
      DoSafeMemCopy(lu_ori_row, lu_col_ * sizeof(T), lu_trans_row, lu_col_ * sizeof(T));

      // update pivot map
      pivots_map[key] = {index, true};
      // put ori data which stored in workspace to mapped new place
      is_visited = pivots_map[index].second;
      while (!is_visited) {
        key = index;
        index = pivots_map[key].first;
        is_visited = pivots_map[key].second;
        lu_ori_row = lu_value + IntToSize(index) * lu_col_;
        T *tmp_wk = lu_trans_wk;
        lu_trans_wk = lu_ori_wk;
        lu_ori_wk = tmp_wk;
        // copy new ori data to trans workspace
        DoSafeMemCopy(lu_trans_wk, lu_col_ * sizeof(T), lu_ori_row, lu_col_ * sizeof(T));
        // copy new data to ori data place
        DoSafeMemCopy(lu_ori_row, lu_col_ * sizeof(T), lu_ori_wk, lu_col_ * sizeof(T));
        pivots_map[key] = {index, true};
      }
    }
    // 4. calculate final permutation matrix
    // for PA = LU get: base + row * permutation_row_ + col
    // for A = PLU  get: base + col * permutation_row_ + row
    // here, we do A = PLU which is same as scipy.
    size_t count = permutation_col_ * permutation_row_ * sizeof(int);
    if (memset_s(reinterpret_cast<void *>(permutation_value), count, 0, count) != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' It does memset_s failed.";
    }
    for (size_t i = 0; i < pivots_col_; ++i) {
      int position = per_value[i];
      int *per_addr = permutation_value + position * SizeToInt(permutation_row_) + SizeToInt(i);
      *per_addr = 1;
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, LUCpuKernelMod::LUFunc>> LUCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LUCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &LUCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> LUCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LUFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LU, LUCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
