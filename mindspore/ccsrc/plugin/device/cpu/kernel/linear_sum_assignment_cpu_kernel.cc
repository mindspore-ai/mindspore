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

#include "plugin/device/cpu/kernel/linear_sum_assignment_cpu_kernel.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include "mindspore/core/ops/linear_sum_assignment.h"

namespace mindspore {
namespace kernel {
namespace {
using LSAP_FUNC_VECTOR = std::vector<std::pair<KernelAttr, LinearSumAssignmentCpuKernelMod::KernelRunFunc>>;

template <typename T>
bool EqualWithPositiveInf(T a, T b) {
  if (std::isinf(a) && std::isinf(b) && a > 0 && b > 0) {
    return true;
  }
  if (std::fabs(a - b) < std::numeric_limits<T>::epsilon()) {
    return true;
  }
  return false;
}

template <typename T>
inline bool check_value(const T *cost, int64_t nr, int64_t nc) {
  for (int64_t i = 0; i < nr * nc; i++) {
    if (std::isnan(cost[i]) || cost[i] == -std::numeric_limits<T>::infinity()) {
      return false;
    }
  }
  return true;
}
}  // namespace

bool LinearSumAssignmentCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (kernel_name_ != prim::kPrimLinearSumAssignment->name()) {
    MS_LOG(ERROR) << "For 'LinearSumAssignment', the kernel name must be 'LinearSumAssignment', but got "
                  << kernel_name_;
    return false;
  }
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int LinearSumAssignmentCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    MS_LOG(ERROR) << kernel_name_ << "resized failed";
    return ret;
  }

  cost_matrix_shape_ = inputs[kIndex0]->GetShapeVector();

  auto cost_matrix_rank = cost_matrix_shape_.size();
  constexpr int64_t kNumber2 = 2;
  if (cost_matrix_rank != kNumber2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the rank of 'cost_matrix' must be 2, but got: " << cost_matrix_rank
                  << ".";
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}

const LSAP_FUNC_VECTOR &LinearSumAssignmentCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, LinearSumAssignmentCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &LinearSumAssignmentCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &LinearSumAssignmentCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

template <typename T>
bool LinearSumAssignmentCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &,
                                                   const std::vector<kernel::AddressPtr> &outputs) {
  int64_t dimension_limit = *reinterpret_cast<int64_t *>(inputs[1]->addr);
  int64_t nr = cost_matrix_shape_[0];
  int64_t nc = cost_matrix_shape_[1];
  if (dimension_limit == INT64_MAX) {
    dimension_limit = nc;
  } else if (dimension_limit > cost_matrix_shape_[1]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "' input dimension_limit must <= the 1st dimension's size of the cost_matrix, "
                      << "which is " << cost_matrix_shape_[1] << ", but got " << dimension_limit << ".";
  } else if (dimension_limit <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input dimension_limit must be positive.";
    return false;
  }

  auto cost_matrix = reinterpret_cast<T *>(inputs[0]->addr);
  auto maximize = reinterpret_cast<bool *>(inputs[2]->addr);
  auto row_ind = reinterpret_cast<int64_t *>(outputs[0]->addr);
  auto col_ind = reinterpret_cast<int64_t *>(outputs[1]->addr);
  if (!Solve(nr, dimension_limit, nc, cost_matrix, *maximize, row_ind, col_ind)) {
    MS_LOG(ERROR) << "Solve linear sum assignment problem failed.";
    return false;
  }
  return true;
}

template <typename T>
int64_t LinearSumAssignmentCpuKernelMod::AugmentingPath(int64_t nc, const T *cost, std::vector<T> *u, std::vector<T> *v,
                                                        std::vector<int64_t> *path, std::vector<int64_t> *row4col,
                                                        std::vector<T> *shortest_path_costs, int64_t i,
                                                        std::vector<bool> *SR, std::vector<bool> *SC,
                                                        std::vector<int64_t> *remaining, T *p_min_val) const {
  size_t num_remaining = LongToSize(nc);
  for (int64_t it = 0; it < nc; it++) {
    remaining->at(LongToSize(it)) = nc - it - 1;
  }

  std::fill(SR->begin(), SR->end(), false);
  std::fill(SC->begin(), SC->end(), false);
  std::fill(shortest_path_costs->begin(), shortest_path_costs->end(), std::numeric_limits<T>::infinity());

  int64_t sink = -1;
  T min_val = 0;
  while (sink == -1) {
    size_t index = 0;
    T lowest = std::numeric_limits<T>::infinity();
    SR->at(LongToSize(i)) = true;

    for (size_t it = 0; it < num_remaining; it++) {
      size_t j = LongToSize(remaining->at(it));

      T r = min_val + cost[LongToSize(i * nc) + j] - u->at(LongToSize(i)) - v->at(j);
      if (r < shortest_path_costs->at(j)) {
        path->at(j) = i;
        shortest_path_costs->at(j) = r;
      }

      if (shortest_path_costs->at(j) < lowest ||
          (EqualWithPositiveInf(shortest_path_costs->at(j), lowest) && row4col->at(j) == -1)) {
        lowest = shortest_path_costs->at(j);
        index = it;
      }
    }

    min_val = lowest;
    if (min_val == std::numeric_limits<T>::infinity()) {
      return -1;
    }

    size_t j = LongToSize(remaining->at(index));
    if (row4col->at(j) == -1) {
      sink = SizeToLong(j);
    } else {
      i = row4col->at(j);
    }

    SC->at(j) = true;
    remaining->at(index) = remaining->at(--num_remaining);
  }

  *p_min_val = min_val;
  return sink;
}

template <typename T>
bool LinearSumAssignmentCpuKernelMod::Solve(int64_t nr, int64_t nc, int64_t raw_nc, T *cost, bool maximize, int64_t *a,
                                            int64_t *b) const {
  if (nr == 0 || nc == 0) {
    return true;
  }

  int64_t element_num = std::min(nr, raw_nc);

  bool transpose = nc < nr;

  std::vector<T> temp;
  if (transpose || maximize) {
    temp.resize(LongToSize(nr * nc));
    ReArrange(&nr, &nc, raw_nc, &temp, cost, transpose, maximize);
    cost = temp.data();
  }

  if (!check_value(cost, nr, nc)) {
    return false;
  }

  std::vector<T> u(nr, 0);
  std::vector<T> v(nc, 0);
  std::vector<T> shortest_path_costs(nc);
  std::vector<int64_t> path(nc, -1);
  std::vector<int64_t> col4row(nr, -1);
  std::vector<int64_t> row4col(nc, -1);
  std::vector<bool> SR(nr);
  std::vector<bool> SC(nc);
  std::vector<int64_t> remaining(nc);

  for (size_t cur_row = 0; cur_row < LongToSize(nr); cur_row++) {
    T min_val;
    int64_t sink = AugmentingPath<T>(nc, cost, &u, &v, &path, &row4col, &shortest_path_costs, SizeToLong(cur_row), &SR,
                                     &SC, &remaining, &min_val);
    if (sink < 0) {
      return false;
    }

    u[cur_row] += min_val;
    for (size_t i = 0; i < LongToSize(nr); i++) {
      if (SR[i] && i != cur_row) {
        u[i] += min_val - shortest_path_costs[LongToSize(col4row[i])];
      }
    }

    for (size_t j = 0; j < LongToSize(nc); j++) {
      if (SC[j]) {
        v[j] -= min_val - shortest_path_costs[j];
      }
    }

    AugmentPreviousSolution(sink, SizeToLong(cur_row), &path, &row4col, &col4row);
  }

  PostProcess(a, b, col4row, transpose, nr, nc, element_num);

  return true;
}

template <typename T>
void LinearSumAssignmentCpuKernelMod::ReArrange(int64_t *origin_nr, int64_t *origin_nc, int64_t raw_nc,
                                                std::vector<T> *temp, const T *cost, bool transpose,
                                                bool maximize) const {
  int64_t nr = *origin_nr;
  int64_t nc = *origin_nc;
  if (transpose) {
    for (int64_t i = 0; i < nr; i++) {
      for (int64_t j = 0; j < nc; j++) {
        temp->at(LongToSize(j * nr + i)) = cost[i * raw_nc + j];
      }
    }
    std::swap(*origin_nr, *origin_nc);
  } else {
    for (int64_t i = 0; i < nr; i++) {
      for (int64_t j = 0; j < nc; j++) {
        temp->at(LongToSize(i * nr + j)) = cost[i * raw_nc + j];
      }
    }
  }

  if (maximize) {
    (void)std::transform(temp->cbegin(), temp->cend(), temp->begin(), [](T value) { return -value; });
  }
}

void LinearSumAssignmentCpuKernelMod::AugmentPreviousSolution(int64_t j, int64_t cur_row, std::vector<int64_t> *path,
                                                              std::vector<int64_t> *row4col,
                                                              std::vector<int64_t> *col4row) const {
  while (true) {
    int64_t i = path->at(LongToSize(j));
    row4col->at(LongToSize(j)) = i;
    std::swap(col4row->at(LongToSize(i)), j);
    if (i == cur_row) {
      break;
    }
  }
}

void LinearSumAssignmentCpuKernelMod::PostProcess(int64_t *a, int64_t *b, const std::vector<int64_t> &col4row,
                                                  bool transpose, int64_t nr, int64_t nc, int64_t element_num) const {
  std::vector<size_t> index(col4row.size());
  std::iota(index.begin(), index.end(), size_t(0));
  std::sort(index.begin(), index.end(), [&col4row](size_t i, size_t j) { return col4row[i] < col4row[j]; });

  if (transpose) {
    size_t i = 0;
    for (auto val : index) {
      a[i] = col4row[val];
      b[i] = SizeToLong(val);
      i++;
    }
  } else {
    for (size_t i = 0; i < LongToSize(nr); i++) {
      a[i] = SizeToLong(i);
      b[i] = col4row[i];
    }
  }

  size_t offset = LongToSize(std::min(nr, nc));
  for (size_t i = offset; i < LongToSize(element_num); i++) {
    a[i] = -1;
    b[i] = -1;
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LinearSumAssignment, LinearSumAssignmentCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
