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

#include "backend/kernel_compiler/cpu/eigen/lu_cpu_kernel.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <unordered_map>
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

template <typename T>
void LUCPUKernel<T>::InitMatrixInfo(const std::vector<size_t> &shape, size_t *row, size_t *col) {
  constexpr size_t lu_min_dim = 1;
  constexpr size_t lu_max_dim = 3;
  if (shape.size() < lu_min_dim || shape.size() > lu_max_dim) {
    MS_LOG_EXCEPTION << kernel_name_ << "shape is " << shape.size() << " which is invalid.";
  }
  if (shape.size() == lu_max_dim) {
    batch_ = shape.front();
    *row = shape.at(lu_min_dim);
    *col = shape.at(lu_max_dim - 1);
    return;
  }
  batch_ = 1;
  *row = shape.front();
  *col = shape.at(lu_min_dim);
}

template <typename T>
void LUCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kLUInputsNum, kernel_name_);
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kLUOutputsNum, kernel_name_);
  auto a_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kLUaIndex);
  InitMatrixInfo(a_shape, &a_row_, &a_col_);
  auto lu_shape = AnfAlgo::GetOutputInferShape(kernel_node, kLuIndex);
  InitMatrixInfo(lu_shape, &lu_row_, &lu_col_);
  auto pivots_shape = AnfAlgo::GetOutputInferShape(kernel_node, kPivotsIndex);
  InitMatrixInfo(pivots_shape, &pivots_row_, &pivots_col_);
  auto permutation_shape = AnfAlgo::GetOutputInferShape(kernel_node, kPermutationIndex);
  InitMatrixInfo(permutation_shape, &permutation_row_, &permutation_col_);
}

template <typename T>
void LUCPUKernel<T>::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  size_t lu_size = lu_col_ * sizeof(T);
  (void)workspace_size_list_.emplace_back(lu_size);
  (void)workspace_size_list_.emplace_back(lu_size);
}

template <typename T>
T LUCPUKernel<T>::GetPermutatedValue(const T *lu_value, const std::vector<int> &per_value, size_t i, size_t j) {
  const T *pered_lu_value = lu_value + per_value[i] * lu_col_ + j;
  return *pered_lu_value;
}

template <typename T>
bool LUCPUKernel<T>::UpdateMajorPermutation(T *lu_value, std::vector<int> *const per_value, int *pivots, size_t k,
                                            size_t rows) {
  T max_major_value = static_cast<T>(kZeroThreshold);
  int max_major_index = 0;
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
  pivots[k] = max_major_index;
  return max_major_value != static_cast<T>(kZeroThreshold);
}

template <typename T>
void LUCPUKernel<T>::SetPermutatedValue(T *lu_value, const std::vector<int> &per_value, size_t i, size_t j,
                                        const T &value) {
  T *per_lu_value = lu_value + per_value[i] * lu_col_ + j;
  *per_lu_value = value;
}

template <typename T>
bool LUCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                            const std::vector<kernel::AddressPtr> &workspace,
                            const std::vector<kernel::AddressPtr> &outputs) {
  // input matrix of (m,n) PA = LU
  T *batch_a_value = reinterpret_cast<T *>(inputs[kLUaIndex]->addr);
  T *batch_lu_value = reinterpret_cast<T *>(outputs[kLuIndex]->addr);
  batch_pivots_ = reinterpret_cast<int *>(outputs[kPivotsIndex]->addr);
  int *batch_permutation_value = reinterpret_cast<int *>(outputs[kPermutationIndex]->addr);
  T *lu_ori_wk = reinterpret_cast<T *>(workspace[kLuIndex]->addr);
  T *lu_trans_wk = reinterpret_cast<T *>(workspace[kPivotsIndex]->addr);
  for (size_t batch = 0; batch < batch_; ++batch) {
    T *a_value = batch_a_value + batch * a_row_ * a_col_;
    T *lu_value = batch_lu_value + batch * lu_row_ * lu_col_;
    // pivots permutation value
    int *pivots = batch_pivots_ + batch * pivots_row_ * pivots_col_;
    // permutation matrix value
    int *permutation_value = batch_permutation_value + batch * permutation_row_ * permutation_col_;
    // init pivots
    std::vector<int> per_value(pivots_col_, 0);
    for (size_t i = 0; i < pivots_col_; ++i) {
      per_value[i] = i;
    }
    // 1. memcpy input to output, do full lu inplace.
    (void)memcpy_s(lu_value, lu_row_ * lu_col_ * sizeof(T), a_value, a_row_ * a_col_ * sizeof(T));

    int s = std::min(a_row_, a_col_);
    // 2. do lu decompose inplace
    for (int k = 0; k < s; ++k) {
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
    for (int i = 0; i < static_cast<int>(lu_row_); ++i) {
      pivots_map[per_value[i]] = {i, false};
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

      T *lu_ori_row = lu_value + index * lu_col_;
      T *lu_trans_row = lu_value + key * lu_col_;
      // copy ori data to trans lu
      (void)memcpy_s(lu_trans_wk, lu_col_ * sizeof(T), lu_ori_row, lu_col_ * sizeof(T));
      // copy new data to ori data ptr
      (void)memcpy_s(lu_ori_row, lu_col_ * sizeof(T), lu_trans_row, lu_col_ * sizeof(T));
      // update pivot map
      pivots_map[key] = {index, true};
      // put ori data which stored in workspace to mapped new place
      is_visited = pivots_map[index].second;
      while (!is_visited) {
        key = index;
        index = pivots_map[key].first;
        is_visited = pivots_map[key].second;
        lu_ori_row = lu_value + index * lu_col_;
        T *tmp_wk = lu_trans_wk;
        lu_trans_wk = lu_ori_wk;
        lu_ori_wk = tmp_wk;
        // copy new ori data to trans workspace
        (void)memcpy_s(lu_trans_wk, lu_col_ * sizeof(T), lu_ori_row, lu_col_ * sizeof(T));
        // copy new data to ori data place
        (void)memcpy_s(lu_ori_row, lu_col_ * sizeof(T), lu_ori_wk, lu_col_ * sizeof(T));
        pivots_map[key] = {index, true};
      }
    }

    // 4. calculate final permutation matrix
    // for PA = LU get: base + row * permutation_row_ + col
    // for A = PLU  get: base + col * permutation_row_ + row
    // here, we do A = PLU which is same as scipy.
    size_t count = permutation_col_ * permutation_row_ * sizeof(int);
    (void)memset_s(reinterpret_cast<void *>(permutation_value), count, 0, count);
    for (size_t i = 0; i < pivots_col_; ++i) {
      int position = per_value[i];
      int *per_addr = permutation_value + position * permutation_row_ + i;
      *per_addr = 1;
    }
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
