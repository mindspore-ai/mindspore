/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "ms_kernel/quantile.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <utility>

#include "common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kQuantileInputNum = 2;
constexpr uint32_t kQuantileOutputNum = 1;
const int64_t paralled_data_size = 64 * 1024;
const int64_t kQuantileAttrDefaultDim = 10000;
const char *kQuantile = "Quantile";
}  // namespace

namespace aicpu {
template <typename T>
uint32_t QuantileCpuKernel::GetInputAndCheck(const CpuKernelContext &ctx) {
  input_ = ctx.Input(0);
  DataType input_type = input_->GetDataType();
  int64_t input_dim = input_->GetTensorShape()->GetDims();
  int64_t input_size = input_->GetTensorShape()->NumElements();
  q_ = ctx.Input(1);
  int64_t q_size = q_->GetTensorShape()->NumElements();
  T *q_addrs = reinterpret_cast<T *>(q_->GetData());
  DataType q_type = q_->GetDataType();
  int64_t q_dim = q_->GetTensorShape()->GetDims();
  int64_t min = -input_dim;
  int64_t max = input_dim - 1;
  auto dim_attr = ctx.GetAttr("dim");
  dim_ = (dim_attr == nullptr) ? kQuantileAttrDefaultDim : dim_attr->GetInt();
  auto keep_dims_attr = ctx.GetAttr("keep_dims");
  keep_dims_ = (keep_dims_attr == nullptr) ? false : keep_dims_attr->GetBool();
  auto ignore_attr = ctx.GetAttr("ignore_nan");
  ignore_nan_ = (ignore_attr == nullptr) ? false : ignore_attr->GetBool();

  KERNEL_CHECK_FALSE(input_size > 0, KERNEL_STATUS_PARAM_INVALID, "quantile() input tensor must be non-empty");
  KERNEL_CHECK_FALSE(q_dim <= 1, KERNEL_STATUS_PARAM_INVALID,
                     "quantile() q must be a scalar or 1D tensor,but got dimension = [%d].", q_dim);
  KERNEL_CHECK_FALSE(input_type == q_type, KERNEL_STATUS_PARAM_INVALID,
                     "quantile() q tensor must be same dtype as the input tensor");

  for (int64_t j = 0; j < q_size; ++j) {
    KERNEL_CHECK_FALSE(q_addrs[j] <= 1 && q_addrs[j] >= 0, KERNEL_STATUS_PARAM_INVALID,
                       "quantile() q values must be in the range [0, 1]");
  }
  DataType out_type = ctx.Output(0)->GetDataType();
  output_ = ctx.Output(0);
  KERNEL_CHECK_FALSE(out_type == input_type, KERNEL_STATUS_PARAM_INVALID,
                     "quantile() out tensor must be same dtype as the input tensor");
  if (dim_ != kQuantileAttrDefaultDim) {
    KERNEL_CHECK_FALSE(dim_ >= min && dim_ <= max, KERNEL_STATUS_PARAM_INVALID,
                       "Dimension out of range (expected to be in range of [%d] and [%d]).", min, max);
  }
  dim_ = MaybeWrapDim(dim_, input_dim);
  return KERNEL_STATUS_OK;
}

uint32_t QuantileCpuKernel::MaybeWrapDim(int64_t dim, int64_t dim_post_expr) {
  if (dim == kQuantileAttrDefaultDim) {
    return dim;
  }
  if (dim_post_expr <= 0) {
    dim_post_expr = 1;
  }
  int64_t min = -dim_post_expr;
  int64_t max = dim_post_expr - 1;
  KERNEL_CHECK_FALSE(dim >= min && dim <= max, KERNEL_STATUS_PARAM_INVALID,
                     "Dimension out of range (expected to be in range of [%d] and [%d]).", min, max)
  if (dim < 0) {
    dim += dim_post_expr;
  }
  return dim;
}

template <typename T>
std::vector<T> transpose(const std::vector<T> &f, const std::vector<int64_t> &shape, int index) {
  int element_count = f.size();
  int m = shape.size();
  int *indexA = reinterpret_cast<int *>(malloc(sizeof(int) * m));
  if (indexA == nullptr) {
    return {};
  }

  std::vector<int> pos(m);
  for (int i = 0; i < m; i++) pos[i] = i;
  if (m != 0) {
    std::swap(pos[m - 1], pos[((index + m) % m)]);
  }

  int *indexB = reinterpret_cast<int *>(malloc(sizeof(int) * m));
  if (indexB == nullptr) {
    free(indexA);
    return {};
  }

  std::vector<T> b(element_count);
  std::vector<int64_t> shapeb(shape);
  for (int i = 0; i < m; i++) {
    shapeb[i] = shape[pos[i]];
  }

  for (int src = 0; src < element_count; src++) {
    int temp = src;
    for (int i = m - 1; i >= 0; i--) {
      indexA[i] = temp % shape[i];
      temp = temp / shape[i];
    }

    for (int i = 0; i < m; i++) {
      indexB[i] = indexA[pos[i]];
    }

    int dst = 0;
    temp = 1;
    for (int i = m - 1; i >= 0; i--) {
      dst = dst + indexB[i] * temp;
      temp = temp * shapeb[i];
    }
    b[dst] = f[src];
  }
  free(indexA);
  free(indexB);

  return b;
}

template <typename T>
void QuantileCpuKernel::QuantileComputeParallelFunc(size_t start, size_t end, int64_t last_shape_size,
                                                    std::vector<T> *sorted) {
  uint64_t q_size = q_->GetTensorShape()->NumElements();
  T *output_addr = reinterpret_cast<T *>(output_->GetData());
  T *q_addrs = reinterpret_cast<T *>(q_->GetData());
  for (u_int64_t i = start; i < end; i++) {
    std::vector<T> tmp;
    std::sort(sorted->begin() + i * last_shape_size, sorted->begin() + (i + 1) * last_shape_size);
    bool has_nan = false;
    bool all_nan = true;

    for (u_int64_t j = i * last_shape_size; j < (i + 1) * last_shape_size; j++) {
      if (std::isnan((*sorted)[j])) {
        has_nan = true;
      } else {
        all_nan = false;
      }
    }

    if ((has_nan && !ignore_nan_) || all_nan) {
      for (uint64_t j = 0; j < q_size; ++j) {
        output_addr[i * q_size + j] = NAN;
      }
      continue;
    }
    for (auto k = i * last_shape_size; k < (i + 1) * last_shape_size; k++) {
      auto x = (*sorted)[k];
      if (!isnan(x)) {
        tmp.push_back(x);
      }
    }
    std::sort(tmp.begin(), tmp.end());
    for (uint64_t j = 0; j < q_size; ++j) {
      T index = (tmp.size() - 1) * q_addrs[j];
      int32_t idx = index;
      if (idx == (int32_t)tmp.size() - 1) {
        output_addr[i * q_size + j] = tmp[idx];
        continue;
      }
      output_addr[i * q_size + j] = tmp[idx] + (tmp[idx + 1] - tmp[idx]) * (index - idx);
    }
  }
}

template <typename T>
void QuantileCpuKernel::QuantileComputeSerialFunc(int64_t last_shape_size, std::vector<T> *sorted) {
  uint64_t n = input_->GetTensorShape()->NumElements();
  uint64_t q_size = q_->GetTensorShape()->NumElements();
  T *output_addr = reinterpret_cast<T *>(output_->GetData());
  T *q_addrs = reinterpret_cast<T *>(q_->GetData());
  for (u_int64_t i = 0; i < n; i += last_shape_size) {
    std::vector<T> tmp;
    sort(sorted->begin() + i, sorted->begin() + i + last_shape_size);
    bool has_nan = false;
    bool all_nan = true;
    for (auto j = i; j < i + last_shape_size; j++) {
      if (!isnan((*sorted)[j])) {
        tmp.push_back((*sorted)[j]);
        all_nan = false;
      } else {
        has_nan = true;
      }
    }
    sort(tmp.begin(), tmp.end());
    for (uint64_t j = 0; j < q_size; ++j) {
      if ((has_nan && !ignore_nan_) || all_nan) {
        output_addr[i * q_size / last_shape_size + j] = NAN;
        continue;
      }

      T index = (tmp.size() - 1) * q_addrs[j];
      int32_t idx = index;
      if (idx == (int32_t)tmp.size() - 1) {
        output_addr[i * q_size / last_shape_size + j] = tmp[idx];
        continue;
      }
      output_addr[i * q_size / last_shape_size + j] = tmp[idx] + (tmp[idx + 1] - tmp[idx]) * (index - idx);
    }
  }
}
template <typename T>
void QuantileCpuKernel::QuantileComputeDefaultFunc(std::vector<T> *sorted) {
  uint64_t q_size = q_->GetTensorShape()->NumElements();
  T *output_addr = reinterpret_cast<T *>(output_->GetData());
  T *q_addrs = reinterpret_cast<T *>(q_->GetData());
  std::sort(sorted->begin(), sorted->end());
  bool all_nan = true;
  std::vector<T> tmp;
  for (auto &x : *sorted) {
    if (!isnan(x)) {
      tmp.push_back(x);
      all_nan = false;
    }
  }
  std::sort(tmp.begin(), tmp.end());
  for (uint64_t i = 0; i < q_size; ++i) {
    if ((has_nan_ && !ignore_nan_) || all_nan) {
      output_addr[i] = NAN;
      continue;
    }
    T index = (tmp.size() - 1) * q_addrs[i];
    int32_t idx = index;
    if (idx == (int32_t)tmp.size() - 1) {
      output_addr[i] = tmp[idx];
      continue;
    }
    output_addr[i] = tmp[idx] + (tmp[idx + 1] - tmp[idx]) * (index - idx);
  }
}

std::vector<int64_t> QuantileCpuKernel::SetQuantileOutputShape() {
  std::vector<int64_t> out_shape;
  int64_t q_dim = q_->GetTensorShape()->NumElements();
  int64_t input_dim = input_->GetTensorShape()->GetDims();
  uint64_t q_size = q_->GetTensorShape()->NumElements();
  std::vector<int64_t> input_shapesize = input_->GetTensorShape()->GetDimSizes();
  if (dim_ != kQuantileAttrDefaultDim && input_dim > 0) {
    out_shape = input_shapesize;
    if (keep_dims_) {
      out_shape[dim_] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim_);
    }
  } else if (keep_dims_) {
    out_shape = std::vector<int64_t>(input_dim, 1);
  }
  if (q_dim > 0) {
    out_shape.insert(out_shape.begin(), q_size);
  }
  return out_shape;
}

template <typename T>
uint32_t QuantileCpuKernel::QuantileCompute(const CpuKernelContext &ctx) {
  T *input_addrs = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  size_t data_size = input_->GetTensorShape()->NumElements() * sizeof(T);

  std::vector<int64_t> out_shape = SetQuantileOutputShape();
  std::vector<int64_t> input_dims = input_->GetTensorShape()->GetDimSizes();
  int64_t input_shape_size = input_->GetTensorShape()->GetDims();
  std::vector<T> sorted;
  int64_t n = input_->GetTensorShape()->NumElements();
  for (int64_t i = 0; i < n; i++) {
    sorted.push_back(input_addrs[i]);
    if (isnan(input_addrs[i])) {
      has_nan_ = true;
    }
  }

  if (data_size <= paralled_data_size) {
    if (dim_ == kQuantileAttrDefaultDim) {
      QuantileComputeDefaultFunc<T>(&sorted);
    } else if (dim_ == input_shape_size - 1) {
      QuantileComputeSerialFunc<T>(input_dims[input_dims.size() - 1], &sorted);
    } else {
      input_dims.push_back(1);
      sorted = transpose<T>(sorted, input_dims, dim_);
      int32_t m = input_dims.size();
      if (m != 0) {
        std::swap(input_dims[m - 1], input_dims[((dim_ + m) % m)]);
      }
      QuantileComputeSerialFunc<T>(input_dims[input_dims.size() - 1], &sorted);
    }
  } else {
    DoParallelQuantile(ctx, sorted, input_dims);
  }
  SetOutput<T>(&out_shape);
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t QuantileCpuKernel::DoParallelQuantile(const CpuKernelContext &ctx, std::vector<T> sorted,
                                               std::vector<int64_t> input_dims) {
  int64_t input_shape_size = input_->GetTensorShape()->GetDims();
  std::vector<int64_t> input_shape_dims = input_->GetTensorShape()->GetDimSizes();
  int64_t n = input_->GetTensorShape()->NumElements();
  if (dim_ == kQuantileAttrDefaultDim) {
    QuantileComputeDefaultFunc<T>(&sorted);
  } else if (dim_ == input_shape_size - 1) {
    int64_t last_shape_size = input_dims[input_dims.size() - 1];
    auto shard_quantile = [&](size_t start, size_t end) {
      QuantileComputeParallelFunc<T>(start, end, last_shape_size, &sorted);
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, n / last_shape_size, last_shape_size, shard_quantile),
                        "Quantile Compute failed.");
  } else {
    input_shape_dims.push_back(1);
    sorted = transpose<T>(sorted, input_shape_dims, dim_);
    int32_t m = input_shape_dims.size();
    if (m != 0) {
      std::swap(input_shape_dims[m - 1], input_shape_dims[((dim_ + m) % m)]);
    }
    int64_t last_shape_size = input_shape_dims[input_shape_dims.size() - 1];
    auto shard_quantile = [&](size_t start, size_t end) {
      QuantileComputeParallelFunc<T>(start, end, last_shape_size, &sorted);
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, n / last_shape_size, last_shape_size, shard_quantile),
                        "Quantile Compute failed.");
  }
  return 0;
}
template <typename T>
void QuantileCpuKernel::SetOutput(std::vector<int64_t> *out_shape) {
  auto output_addr = reinterpret_cast<T *>(output_->GetData());

  int64_t l = output_->GetTensorShape()->NumElements();
  std::vector<T> out;
  int64_t q_dim = q_->GetTensorShape()->GetDims();
  std::vector<int64_t> tmp(*out_shape);
  if (q_dim > 0) {
    for (int i = 0; i < l; i++) {
      out.push_back(*(output_addr + i));
    }

    int64_t out_end_shape = (*out_shape)[out_shape->size() - 1];
    out_shape->push_back(out_end_shape);
    std::swap((*out_shape)[0], (*out_shape)[out_shape->size() - 1]);
    out_shape->erase(out_shape->begin());
    out_shape->insert(out_shape->begin(), 1);
    out = transpose<T>(out, *out_shape, 0);
    for (int i = 0; i < l; i++) {
      output_addr[i] = out[i];
    }
  }
  output_->GetTensorShape()->SetDimSizes(tmp);
}

uint32_t QuantileCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kQuantileInputNum, kQuantileOutputNum), "[%s] check params failed.", kQuantile);
  uint32_t res = KERNEL_STATUS_OK;

  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      res = GetInputAndCheck<float>(ctx);
      break;
    case DT_DOUBLE:
      res = GetInputAndCheck<double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Quantile invalid input type [%s]", DTypeStr(data_type).c_str());
      break;
  }
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "GetInputAndCheck failed.");
  switch (data_type) {
    case DT_FLOAT:
      res = QuantileCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      res = QuantileCompute<double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Quantile invalid input type [%s]", DTypeStr(data_type).c_str());
      break;
  }
  if (res != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kQuantile, QuantileCpuKernel);
}  // namespace aicpu
