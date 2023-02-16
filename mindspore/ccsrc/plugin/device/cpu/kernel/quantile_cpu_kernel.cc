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
#include <iostream>
#include <cmath>
#include "plugin/device/cpu/kernel/quantile_cpu_kernel.h"
#include "mindspore/core/ops/quantile.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kQuantileInputsNum = 2;
constexpr size_t kQuantileOutputsNum = 1;
constexpr int kQuantileDefaultDim = 10000;
}  // namespace

bool QuantileCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Quantile>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast Quantile ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  dim_ = GetValue<int64_t>(base_operator->GetAttr("dim"));
  keep_dims_ = GetValue<bool>(base_operator->GetAttr("keep_dims"));
  ignore_nan_ = GetValue<bool>(base_operator->GetAttr("ignore_nan"));
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int QuantileCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != KRET_OK) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return ret;
  }
  input_shape_ = inputs.at(0)->GetShapeVector();
  q_shape_ = inputs.at(1)->GetShapeVector();
  return KRET_OK;
}

uint32_t QuantileCpuKernelMod::MaybeWrapDim(int dim, int dim_post_expr) {
  if (dim == kQuantileDefaultDim) {
    return dim;
  }
  if (dim_post_expr <= 0) {
    dim_post_expr = 1;
  }
  int min = -dim_post_expr;
  int max = dim_post_expr - 1;
  if (dim < min || dim > max) {
    MS_EXCEPTION(ValueError) << "For 'Quantile', dimension out of range (expected to be in range of " << min
                             << " and [ " << max << "]).";
  }
  if (dim < 0) {
    dim += dim_post_expr;
  }
  return dim;
}

template <typename T>
std::vector<T> transpose(const std::vector<T> &f, const std::vector<int64_t> &shape, int index) {
  size_t element_count = f.size();
  size_t m = SizeToInt(shape.size());
  std::vector<int> pos(m);
  std::vector<int> indexA(m);
  std::vector<int> indexB(m);
  std::vector<T> target(element_count);
  std::vector<int64_t> shapeTarget(shape);

  for (size_t j = 0; j < m; j++) {
    pos[j] = SizeToInt(j);
  }
  if (m != 0) {
    std::swap(pos[m - 1], pos[((index + m) % m)]);
  }

  for (size_t j = 0; j < m; j++) {
    shapeTarget[j] = shape[pos[j]];
  }

  for (size_t src = 0; src < element_count; src++) {
    int temp = src;
    for (int i = SizeToInt(m) - 1; i >= 0; i--) {
      indexA[i] = temp % LongToInt(shape[i]);
      temp = temp / LongToInt(shape[i]);
    }

    for (size_t i = 0; i < m; i++) {
      indexB[i] = indexA[pos[i]];
    }

    int dst = 0;
    temp = 1;
    for (int i = SizeToInt(m) - 1; i >= 0; i--) {
      dst = dst + indexB[i] * temp;
      temp = temp * shapeTarget[i];
    }
    target[dst] = f[src];
  }

  return target;
}

template <typename T>
void QuantileComputeDefaultFunc(uint64_t n, uint64_t q_size, const std::vector<T> &sorted, T *output_addr, T *q_addrs,
                                bool has_nan_, bool ignore_nan_) {
  std::vector<T> tmp(sorted);

  std::sort(tmp.begin(), tmp.end());
  bool all_nan = true;
  tmp.clear();
  for (auto &x : sorted) {
    if (!std::isnan(x)) {
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

    if (idx == SizeToInt(tmp.size()) - 1) {
      output_addr[i] = tmp[idx];
      continue;
    }

    output_addr[i] = tmp[idx] + (tmp[idx + 1] - tmp[idx]) * (index - idx);
  }
}

std::vector<int64_t> SetQuantileOutputShape(int64_t dim, int64_t input_dim, bool keep_dims, const int64_t q_size,
                                            const std::vector<int64_t> &input_shapesize, int64_t q_dim) {
  std::vector<int64_t> out_shape;
  if (dim != kQuantileDefaultDim && input_dim > 0) {
    out_shape = input_shapesize;
    if (keep_dims) {
      out_shape[dim] = 1;
    } else {
      (void)out_shape.erase(out_shape.begin() + dim);
    }
  } else if (keep_dims) {
    out_shape = std::vector<int64_t>(input_dim, 1);
  }
  if (q_dim > 0) {
    (void)out_shape.insert(out_shape.begin(), q_size);
  }
  return out_shape;
}

template <typename T>
bool QuantileCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &workspace,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs[kIndex1]->size == 0) {
    MS_EXCEPTION(ValueError) << "For Quantile, q-th must be non-empty";
  }

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kQuantileInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kQuantileOutputsNum, kernel_name_);

  auto input = static_cast<T *>(inputs[0]->addr);
  auto q = static_cast<T *>(inputs[1]->addr);
  auto output = static_cast<T *>(outputs[0]->addr);

  size_t input_dim = input_shape_.size();
  size_t q_dim = q_shape_.size();
  size_t q_size = inputs[1]->size / sizeof(T);
  total_ = inputs[0]->size / sizeof(T);
  auto output_size = outputs[0]->size / sizeof(T);

  auto input_shape = input_shape_;
  dim_ = static_cast<int64_t>(MaybeWrapDim(dim_, input_shape.size()));
  if (total_ <= 0) {
    MS_EXCEPTION(ValueError) << "For Quantile, input tensor must be non-empty";
  }

  if (q_dim > 1) {
    MS_EXCEPTION(ValueError) << "For 'Quantile', the input q must be a scalar or 1D tensor,but got dimension = "
                             << q_dim << ".";
  }

  for (size_t j = 0; j < q_size; ++j) {
    if (q[j] < 0 || q[j] > 1) {
      MS_EXCEPTION(ValueError) << "For 'Quantile', q values must be in the range [0, 1].";
    }
  }

  auto input_shapesize = input_shape_;
  std::vector<T> sorted;
  std::vector<int64_t> out_shape =
    SetQuantileOutputShape(dim_, input_shape_.size(), keep_dims_, q_size, input_shapesize, q_shape_.size());
  for (uint64_t i = 0; i < total_; i++) {
    sorted.push_back(input[i]);
    if (std::isnan(input[i])) {
      has_nan_ = true;
    }
  }

  if (dim_ == kQuantileDefaultDim) {
    QuantileComputeDefaultFunc<T>(total_, q_size, sorted, output, q, has_nan_, ignore_nan_);
  } else if (dim_ == SizeToInt(input_dim) - 1) {
    int64_t last_shape_size = input_shape[input_shape.size() - 1];
    ParallelRun(last_shape_size, q_size, sorted, output, q);
  } else {
    input_shapesize.push_back(1);
    sorted = transpose<T>(sorted, input_shapesize, dim_);

    int32_t m = SizeToInt(input_shapesize.size());
    if (m != 0) {
      std::swap(input_shapesize[m - 1], input_shapesize[((dim_ + m) % m)]);
    }
    int64_t last_shape_size = input_shapesize[input_shapesize.size() - 1];
    ParallelRun(last_shape_size, q_size, sorted, output, q);
  }

  std::vector<T> out;
  if (q_dim > 0) {
    for (size_t i = 0; i < output_size; i++) {
      out.push_back(*(output + i));
    }
    int64_t out_end_shape = out_shape[out_shape.size() - 1];
    out_shape.push_back(out_end_shape);
    std::swap(out_shape[0], out_shape[out_shape.size() - 1]);
    (void)out_shape.erase(out_shape.begin());
    (void)out_shape.insert(out_shape.begin(), 1);
    out = transpose<T>(out, out_shape, 0);
    for (size_t i = 0; i < output_size; i++) {
      output[i] = out[i];
    }
  }
  return true;
}

template <typename T>
void QuantileCpuKernelMod::ParallelRun(int64_t last_shape_size, uint64_t q_size, const std::vector<T> &sorted,
                                       T *output_addr, T *q_addrs) {
  const int64_t thread_num = SizeToLong(FloatToSize(std::ceil(total_ / last_shape_size)));
  std::vector<common::Task> tasks;
  for (uint64_t task_id = 0; task_id < LongToUlong(thread_num) && task_id * last_shape_size < total_; task_id++) {
    uint64_t start = task_id * LongToUlong(last_shape_size);
    uint64_t end = (task_id + 1) * LongToUlong(last_shape_size);
    auto task = [this, &last_shape_size, &q_size, &sorted, &output_addr, &q_addrs, start, end]() {
      DoQuantile(last_shape_size, q_size, sorted, output_addr, q_addrs, start, end);
      return common::SUCCESS;
    };

    (void)tasks.emplace_back(task);
  }
  ParallelLaunch(tasks);
}

template <typename T>
void QuantileCpuKernelMod::DoQuantile(int64_t last_shape_size, uint64_t q_size, const std::vector<T> &sorted,
                                      T *output_addr, T *q_addrs, uint64_t start, uint64_t end) {
  std::vector<T> tmp;
  bool has_nan = false;
  bool all_nan = true;
  for (auto i = start; i < end; i++) {
    tmp.push_back(sorted[i]);
    if (std::isnan(sorted[i])) {
      has_nan = true;
    } else {
      all_nan = false;
    }
  }
  std::sort(tmp.begin(), tmp.end());

  bool flag = (has_nan && !ignore_nan_) || all_nan;
  if (flag) {
    for (uint64_t j = 0; j < q_size; ++j) {
      output_addr[start / LongToUlong(last_shape_size) * q_size + j] = NAN;
    }
  } else {
    tmp.clear();
    for (auto i = start; i < end; i++) {
      auto x = sorted[i];
      if (!std::isnan(x)) {
        tmp.push_back(x);
      }
    }
    std::sort(tmp.begin(), tmp.end());

    for (uint64_t j = 0; j < q_size; ++j) {
      T index = (tmp.size() - 1) * q_addrs[j];

      int32_t idx = static_cast<int32_t>(index);
      if (idx == SizeToInt(tmp.size()) - 1) {
        output_addr[start / LongToUlong(last_shape_size) * q_size + j] = tmp[idx];
        continue;
      }
      output_addr[start / LongToUlong(last_shape_size) * q_size + j] =
        tmp[idx] + (index - idx) * (tmp[idx + 1] - tmp[idx]);
    }
  }
}

const std::vector<std::pair<KernelAttr, QuantileCpuKernelMod::KernelRunFunc>> &QuantileCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, QuantileCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &QuantileCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &QuantileCpuKernelMod::LaunchKernel<double>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Quantile, QuantileCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
