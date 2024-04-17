/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#include <cmath>
#include <cstring>
#include <numeric>
#include <memory>

#include "linear_sum_assignment.h"
#include "securec.h"
#include "context/common/status.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"

#include "inc/kernel_log.h"

namespace {
const char *const kLinearSumAssignment = "LinearSumAssignment";
constexpr uint32_t kOutputNum = 2;
constexpr uint32_t kInputNum = 3;

template <typename T>
bool EqualWithPositiveInf(T num1, T num2) {
  if (std::isinf(num1) && std::isinf(num2) && num1 > 0 && num2 > 0) {
    return true;
  }

  if (std::fabs(num1 - num2) < std::numeric_limits<T>::epsilon()) {
    return true;
  }

  return false;
}

template <typename T>
inline bool CheckValue(const T *const cost, uint64_t nr, uint64_t nc) {
  for (uint64_t i = 0; i < nr * nc; i++) {
    if (std::isnan(cost[i]) || (std::isinf(cost[i]) && cost[i] < 0)) {
      return false;
    }
  }
  return true;
}

#define LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DTYPE, TYPE, CTX)                    \
  case (DTYPE): {                                                               \
    uint32_t ret = LinearSumAssignmentCompute<TYPE>(CTX);                       \
    if (ret != KERNEL_STATUS_OK) {                                              \
      CUST_KERNEL_LOG_ERROR(CTX, "LinearSumAssignment kernel compute failed."); \
      return ret;                                                               \
    }                                                                           \
    break;                                                                      \
  }

#define LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE_ALL(CTX)                \
  LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DT_BOOL, bool, CTX)           \
  LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DT_INT8, int8_t, CTX)         \
  LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DT_INT16, int16_t, CTX)       \
  LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DT_INT32, int32_t, CTX)       \
  LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DT_INT64, int64_t, CTX)       \
  LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DT_UINT8, uint8_t, CTX)       \
  LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DT_UINT16, uint16_t, CTX)     \
  LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DT_UINT32, uint32_t, CTX)     \
  LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DT_UINT64, uint64_t, CTX)     \
  LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DT_FLOAT16, Eigen::half, CTX) \
  LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DT_FLOAT, float, CTX)         \
  LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE(DT_DOUBLE, double, CTX)

}  // namespace

namespace aicpu {
uint32_t LinearSumAssignmentCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "Check LinearSumAssignment params failed.");
  auto matrix_data_type = ctx.Input(0)->GetDataType();
  auto maximize_data_type = ctx.Input(2)->GetDataType();
  auto row_ind_data_type = ctx.Output(0)->GetDataType();
  auto col_ind_data_type = ctx.Output(1)->GetDataType();
  if (maximize_data_type != DT_BOOL || row_ind_data_type != DT_INT64 || col_ind_data_type != DT_INT64) {
    CUST_KERNEL_LOG_ERROR(
      ctx,
      "[%s] Data type of input is not support, maximize data type is [%s], row_ind data type is [%s], col_ind data "
      "type is [%s].",
      ctx.GetOpType().c_str(), DTypeStr(maximize_data_type).c_str(), DTypeStr(row_ind_data_type).c_str(),
      DTypeStr(col_ind_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (matrix_data_type) {
    LINEAR_SUM_ASSIGNMENT_COMPUTE_CASE_ALL(ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "LinearSumAssignment matrix data type [%s] not support.",
                            DTypeStr(matrix_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LinearSumAssignmentCpuKernel::SolveProblem(CpuKernelContext &ctx, T *cost, int64_t *a, int64_t *b) {
  std::shared_ptr<float> cost_matrix_buf(new float[nr * raw_nc]);
  for (uint64_t i = 0; i < nr; i++) {
    for (uint64_t j = 0; j < raw_nc; j++) {
      *(cost_matrix_buf.get() + i * raw_nc + j) = static_cast<float>(*(cost + i * raw_nc + j));
    }
  }
  return Solve<float>(ctx, cost_matrix_buf.get(), a, b);
}

#define SOLVE_PROBLEM_FLOAT(TYPE)                                                                                  \
  template <>                                                                                                      \
  uint32_t LinearSumAssignmentCpuKernel::SolveProblem(CpuKernelContext &ctx, TYPE *cost, int64_t *a, int64_t *b) { \
    return Solve<TYPE>(ctx, cost, a, b);                                                                           \
  }

#define SOLVE_PROBLEM_FLOAT_ALL()  \
  SOLVE_PROBLEM_FLOAT(Eigen::half) \
  SOLVE_PROBLEM_FLOAT(float)       \
  SOLVE_PROBLEM_FLOAT(double)

SOLVE_PROBLEM_FLOAT_ALL()

template <typename T>
uint32_t LinearSumAssignmentCpuKernel::LinearSumAssignmentCompute(CpuKernelContext &ctx) {
  Tensor *input1 = ctx.Input(0);
  Tensor *input2 = ctx.Input(1);
  Tensor *input3 = ctx.Input(2);
  Tensor *output1 = ctx.Output(0);
  Tensor *output2 = ctx.Output(1);

  auto input_shape = input1->GetTensorShape();
  ShapeVector dim_sizes = input_shape->GetDimSizes();
  if (!IsMatrix(dim_sizes)) {
    CUST_KERNEL_LOG_ERROR(
      ctx, "LinearSumAssignment first input is not a matrix. Expected dim size is 2, but got dim size [%u].",
      dim_sizes.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto nr_signed = static_cast<int64_t>(input_shape->GetDimSize(0));
  auto raw_nc_signed = static_cast<int64_t>(input_shape->GetDimSize(1));

  auto input_cost = reinterpret_cast<T *>(input1->GetData());
  auto nc_signed = *reinterpret_cast<int64_t *>(input2->GetData());
  nc_signed = (nc_signed == INT64_MAX) ? raw_nc_signed : nc_signed;
  maximize = *reinterpret_cast<bool *>(input3->GetData());
  int64_t *a = reinterpret_cast<int64_t *>(output1->GetData());
  int64_t *b = reinterpret_cast<int64_t *>(output2->GetData());

  if (nr_signed < 0 || nc_signed < 0 || nc_signed > raw_nc_signed) {
    CUST_KERNEL_LOG_ERROR(
      ctx,
      "LinearSumAssignment input param dimension is not correct. nr_signed: [%u], nc_signed: [%u], raw_nc_signed: "
      "[%u].",
      nr_signed, nc_signed, raw_nc_signed);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  nr = static_cast<uint64_t>(nr_signed);
  raw_nc = static_cast<uint64_t>(raw_nc_signed);
  nc = static_cast<uint64_t>(nc_signed);

  return SolveProblem<T>(ctx, input_cost, a, b);
}

template <typename T>
uint32_t LinearSumAssignmentCpuKernel::Solve(CpuKernelContext &ctx, const T *cost, int64_t *a, int64_t *b) {
  if (nr == 0 || nc == 0) {
    return 0;
  }

  uint64_t element_num = std::min(nr, raw_nc);
  auto bytes_num = sizeof(int64_t) * element_num;
  (void)memset_s(a, bytes_num, 0, bytes_num);
  (void)memset_s(b, bytes_num, 0, bytes_num);

  std::vector<T> temp;
  bool transpose = nc < nr;
  temp.resize(nr * nc);
  ReArrange(&temp, cost, transpose);
  cost = temp.data();

  if (!CheckValue(cost, nr, nc)) {
    CUST_KERNEL_LOG_ERROR(ctx, "CheckValue Error. cost can\'t be nan or -inf");
    return KERNEL_STATUS_INNER_ERROR;
  }

  T zero(0);
  std::vector<T> u(nr, zero);
  std::vector<T> v(nc, zero);
  std::vector<T> shortestPathCosts(nc);
  path = std::vector<int64_t>(nc, -1);
  col4row = std::vector<int64_t>(nr, -1);
  row4col = std::vector<int64_t>(nc, -1);
  SR = std::vector<bool>(nr);
  SC = std::vector<bool>(nc);
  remaining = std::vector<uint64_t>(nc);

  for (cur_row = 0; cur_row < nr; cur_row++) {
    T minVal;
    int64_t sink = AugmentingPath(cost, u, v, shortestPathCosts, &minVal);
    if (sink < 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "sink can\'t be less than 0. sink value: [%d]", sink);
      return KERNEL_STATUS_INNER_ERROR;
    }

    u[cur_row] += minVal;
    for (uint64_t i = 0; i < nr; i++) {
      if (SR[i] && i != cur_row) {
        uint64_t idx = static_cast<uint64_t>(col4row[i]);
        u[i] += minVal - shortestPathCosts[idx];
      }
    }

    for (uint64_t j = 0; j < nc; j++) {
      if (SC[j]) {
        v[j] -= minVal - shortestPathCosts[j];
      }
    }

    AugmentPreviousSolution(sink);
  }

  PostProcess(a, b, transpose, element_num);

  return KERNEL_STATUS_OK;
}

template <typename T>
int64_t LinearSumAssignmentCpuKernel::AugmentingPath(const T *const cost, std::vector<T> &u, std::vector<T> &v,
                                                     std::vector<T> &shortestPathCosts, T *p_minVal) {
  T minVal(0);
  uint64_t i = cur_row;

  uint64_t num_remaining = nc;
  for (uint64_t it = 0; it < nc; it++) {
    remaining[it] = (nc - it) - 1;
  }

  std::fill(SR.begin(), SR.end(), false);
  std::fill(SC.begin(), SC.end(), false);
  T infinity(std::numeric_limits<T>::infinity());
  std::fill(shortestPathCosts.begin(), shortestPathCosts.end(), infinity);

  int64_t sink = -1;
  while (sink == -1) {
    uint64_t index = 0;
    T lowest(std::numeric_limits<T>::infinity());
    SR[i] = true;

    for (uint64_t it = 0; it < num_remaining; it++) {
      uint64_t j = remaining[it];

      T r((minVal + cost[i * nc + j] - u[i]) - v[j]);
      if (r < shortestPathCosts[j]) {
        path[j] = static_cast<int64_t>(i);
        shortestPathCosts[j] = r;
      }

      if (shortestPathCosts[j] < lowest || (EqualWithPositiveInf(shortestPathCosts[j], lowest) && row4col[j] == -1)) {
        lowest = shortestPathCosts[j];
        index = it;
      }
    }

    minVal = lowest;
    if (std::isinf(minVal) && minVal > 0) {
      return -1;
    }

    uint64_t j = remaining[index];
    if (row4col[j] == -1) {
      sink = static_cast<int64_t>(j);
    } else {
      i = static_cast<uint64_t>(row4col[j]);
    }

    SC[j] = true;
    remaining[index] = remaining[--num_remaining];
  }

  *p_minVal = minVal;
  return sink;
}

template <typename T>
std::vector<uint64_t> LinearSumAssignmentCpuKernel::ArgSortIter(const std::vector<T> &v) {
  std::vector<uint64_t> index(v.size());
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(), [&v](uint64_t i, uint64_t j) { return v[i] < v[j]; });
  return index;
}

template <typename T>
void LinearSumAssignmentCpuKernel::ReArrange(std::vector<T> *temp, const T *const cost, bool transpose) {
  if (transpose) {
    for (uint64_t i = 0; i < nr; i++) {
      for (uint64_t j = 0; j < nc; j++) {
        temp->at(j * nr + i) = cost[i * raw_nc + j];
      }
    }
    std::swap(nr, nc);
  } else {
    for (uint64_t i = 0; i < nr; i++) {
      for (uint64_t j = 0; j < nc; j++) {
        temp->at(i * nc + j) = cost[i * raw_nc + j];
      }
    }
  }

  if (maximize) {
    (void)std::transform(temp->cbegin(), temp->cend(), temp->begin(), [](T value) { return -value; });
  }
}

void LinearSumAssignmentCpuKernel::AugmentPreviousSolution(int64_t j) {
  uint64_t t = static_cast<uint64_t>(j);
  uint64_t i;
  do {
    i = static_cast<uint64_t>(path[t]);
    row4col[t] = path[t];
    int64_t temp = col4row[i];
    col4row[i] = static_cast<int64_t>(t);
    t = static_cast<uint64_t>(temp);
  } while (i != cur_row);
}

void LinearSumAssignmentCpuKernel::PostProcess(int64_t *a, int64_t *b, bool transpose, uint64_t element_num) {
  if (transpose) {
    uint64_t i = 0;
    for (auto val : ArgSortIter(col4row)) {
      a[i] = col4row[val];
      b[i] = static_cast<int64_t>(val);
      i++;
    }
  } else {
    for (uint64_t i = 0; i < nr; i++) {
      a[i] = static_cast<int64_t>(i);
      b[i] = col4row[i];
    }
  }

  uint64_t offset = std::min(nr, nc);
  for (uint64_t i = offset; i < element_num; i++) {
    a[i] = -1;
    b[i] = -1;
  }
}

REGISTER_MS_CPU_KERNEL(kLinearSumAssignment, LinearSumAssignmentCpuKernel);
}  // namespace aicpu
