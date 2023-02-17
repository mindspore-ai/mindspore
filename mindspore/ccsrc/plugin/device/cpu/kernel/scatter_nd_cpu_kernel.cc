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

#include "plugin/device/cpu/kernel/scatter_nd_cpu_kernel.h"
#include <algorithm>
#include <string>
#include <mutex>
#include "include/common/thread_pool.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kScatterNdOutputSize = 1;
constexpr size_t kMinIndiceRank = 2;
constexpr char kKernelName[] = "ScatterNd";

template <typename S, typename T>
struct ComputeParams {
  T *target_{nullptr};
  S *indices_{nullptr};
  T *updates_{nullptr};
  int unit_size_{0};
  int indices_unit_rank_{0};
  std::vector<int> *out_strides_{nullptr};
  size_t target_mem_size_{0};
};

template <typename S, typename T>
void ParallelLaunch(ScatterNdCpuKernelMod *content, const ComputeParams<S, T> *params, size_t offset, size_t unit) {
  T *target = params->target_;
  T *updates = params->updates_;
  auto compute_func = [&target, &updates](size_t t_idx, size_t u_idx) {
    auto &atomic_ = reinterpret_cast<std::atomic<T> *>(target)[t_idx];
    T expect = atomic_.load();
    T result;
    do {
      result = expect + updates[u_idx];
    } while (!atomic_.compare_exchange_weak(expect, result));
  };
  auto task = [&compute_func, &params, offset, unit](size_t update_start, size_t update_end) {
    for (size_t idx = update_start; idx < update_end; idx++) {
      compute_func(offset + idx, unit + idx);
    }
  };
  ParallelLaunchAutoSearch(task, IntToSize(params->unit_size_), content, &content->parallel_search_info_);
}

// the std::atomic don't support complex type, use mutex instead.
template <typename S, typename T>
void LaunchForComplex(ScatterNdCpuKernelMod *content, const ComputeParams<S, T> *params, size_t offset, size_t unit) {
  T *target = params->target_;
  T *updates = params->updates_;
  std::mutex task_mutex;
  auto task = [&target, &updates, unit, offset, &task_mutex](size_t update_start, size_t update_end) {
    for (size_t idx = update_start; idx < update_end; idx++) {
      std::lock_guard<std::mutex> task_lock(task_mutex);
      target[offset + idx] += updates[unit + idx];
    }
  };
  ParallelLaunchAutoSearch(task, IntToSize(params->unit_size_), content, &content->parallel_search_info_);
}

#define DEF_COMPLEX_LAUNCH_FUNC(S, T)                                                                          \
  template <>                                                                                                  \
  void ParallelLaunch<S, T>(ScatterNdCpuKernelMod * content, const ComputeParams<S, T> *params, size_t offset, \
                            size_t unit) {                                                                     \
    LaunchForComplex(content, params, offset, unit);                                                           \
  }

DEF_COMPLEX_LAUNCH_FUNC(int16_t, complex128)
DEF_COMPLEX_LAUNCH_FUNC(int32_t, complex128)
DEF_COMPLEX_LAUNCH_FUNC(int64_t, complex128)
DEF_COMPLEX_LAUNCH_FUNC(int16_t, complex64)
DEF_COMPLEX_LAUNCH_FUNC(int32_t, complex64)
DEF_COMPLEX_LAUNCH_FUNC(int64_t, complex64)

template <typename S, typename T>
void Compute(ScatterNdCpuKernelMod *content, const ComputeParams<S, T> *params, const size_t start, const size_t end) {
  S *indices = params->indices_;
  std::vector<int> *out_strides = params->out_strides_;

  for (size_t i = start; i < end; ++i) {
    int offset = 0;
    auto indices_unit_rank = IntToSize(params->indices_unit_rank_);
    for (size_t j = 0; j < indices_unit_rank; ++j) {
      int index = static_cast<int>(indices[i * indices_unit_rank + j]);
      if (index < 0) {
        MS_LOG(EXCEPTION) << "For '" << kKernelName
                          << "', each element in 'indices' must be greater "
                             "than or equal to 0, but got "
                          << index;
      }
      if (index > static_cast<int>(content->out_shape_[j])) {
        MS_LOG(EXCEPTION) << "For '" << kKernelName
                          << "', each element in 'indices' should be smaller than the value of shape, but got " << index
                          << " and got the value of shape " << content->out_shape_[j];
      }
      offset += index * out_strides->at(j) * params->unit_size_;
    }
    ParallelLaunch(content, params, IntToSize(offset), IntToSize(params->unit_size_) * i);
  }
}
}  // namespace

bool ScatterNdCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t kDynamicInputNum = 3;
  kernel_name_ = base_operator->GetPrim()->name();
  if (inputs.size() != kDynamicInputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 3, but got " << inputs.size()
                  << " input(s).";
    return false;
  }

  constexpr size_t output_num = 1;
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int ScatterNdCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  out_shape_ = outputs[kIndex0]->GetShapeVector();
  auto indices_shape = inputs[kIndex0]->GetShapeVector();
  auto updates_shape = inputs[kIndex1]->GetShapeVector();
  auto indices_unit_rank = indices_shape.back();
  if (indices_unit_rank > static_cast<int64_t>(out_shape_.size())) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the value of last dimension of 'indices' must be less than "
                     "or equal to the dimension of 'shape', but got  the value of last "
                     "dimension of 'indices': "
                  << indices_unit_rank << " and the dimension of 'shape': " << out_shape_.size();
    return KRET_RESIZE_FAILED;
  }
  if (indices_shape.size() < kMinIndiceRank) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'indices' must be at least 2, but got "
                  << indices_shape.size();
    return KRET_RESIZE_FAILED;
  }
  if (updates_shape.size() != indices_shape.size() - 1 + out_shape_.size() - indices_unit_rank) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'update' must be equal to the "
                     "dimension of 'indices' minus 1 plus the "
                     "dimension of 'shape' minus the value of last "
                     "dimension of 'indices', but got "
                  << "the dimension of 'update': " << updates_shape.size() << ", the dimension of 'indices' "
                  << indices_shape.size() << ", the dimension of 'shape' " << out_shape_.size()
                  << ", the value of last dimension of 'indices' " << indices_unit_rank;
    return KRET_RESIZE_FAILED;
  }
  for (size_t i = 0; i < indices_shape.size() - 1; ++i) {
    if (updates_shape[i] != indices_shape[i]) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the shape of 'updates' and 'indices' are "
                       "different in dimension i="
                    << i << ". The 'updates_shape[i]' is " << updates_shape[i] << " and the 'indices_shape[i]' is "
                    << indices_shape[i];
      return KRET_RESIZE_FAILED;
    }
  }
  indices_unit_rank_ = SizeToInt(indices_unit_rank);
  unit_size_ = 1;
  for (size_t i = indices_shape.size() - 1; i < updates_shape.size(); ++i) {
    unit_size_ *= SizeToInt(updates_shape[i]);
  }

  num_units_ = 1;
  const size_t two = 2;
  num_units_ *= LongToSize(updates_shape[indices_shape.size() - two]);
  for (int i = SizeToInt(indices_shape.size()) - 3; i >= 0; i--) {
    num_units_ *= LongToSize(updates_shape[IntToSize(i)]);
  }
  out_strides_.clear();
  int out_stride = 1;
  out_strides_.push_back(out_stride);
  for (int i = indices_unit_rank_ - 2; i >= 0; i--) {
    out_stride *= SizeToInt(out_shape_[IntToSize(i + 1)]);
    out_strides_.push_back(out_stride);
  }
  reverse(out_strides_.begin(), out_strides_.end());
  return KRET_OK;
}

template <typename S, typename T>
bool ScatterNdCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto target = reinterpret_cast<T *>(outputs[0]->addr);
  auto target_init = memset_s(target, outputs[0]->size, 0, outputs[0]->size);
  if (target_init != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset failed. Error no: " << target_init;
  }
  ComputeParams<S, T> params;
  params.target_ = target;
  params.indices_ = reinterpret_cast<S *>(inputs[0]->addr);
  params.updates_ = reinterpret_cast<T *>(inputs[1]->addr);
  params.target_mem_size_ = outputs[0]->size;
  params.unit_size_ = unit_size_;
  params.indices_unit_rank_ = indices_unit_rank_;
  params.out_strides_ = &out_strides_;
  for (size_t idx = 0; idx < num_units_; idx++) {
    Compute<S, T>(this, &params, idx, idx + 1);
  }
  return true;
}

#define DTYPE_REGISTER_(DT1, UPDATES, SHAPE, OUTPUT, DT2, T)                                      \
  KernelAttr().AddInputAttr(DT1).AddInputAttr(UPDATES).AddInputAttr(SHAPE).AddOutputAttr(OUTPUT), \
    &ScatterNdCpuKernelMod::LaunchKernel<DT2, T>

#define DTYPE_REGISTER(UPDATES, SHAPE, OUTPUT, T)                              \
  {DTYPE_REGISTER_(kNumberTypeInt16, UPDATES, SHAPE, OUTPUT, int16_t, T)},     \
    {DTYPE_REGISTER_(kNumberTypeInt32, UPDATES, SHAPE, OUTPUT, int32_t, T)}, { \
    DTYPE_REGISTER_(kNumberTypeInt64, UPDATES, SHAPE, OUTPUT, int64_t, T)      \
  }

std::vector<std::pair<KernelAttr, ScatterNdCpuKernelMod::ScatterNdFunc>> ScatterNdCpuKernelMod::func_list_ = {
  DTYPE_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, double),
  DTYPE_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, float),
  DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, int64_t),
  DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int32_t),
  DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt16, int16_t),
  DTYPE_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, int8_t),
  DTYPE_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeUInt64, uint64_t),
  DTYPE_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeUInt32, uint32_t),
  DTYPE_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeUInt16, uint16_t),
  DTYPE_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, uint8_t),
  DTYPE_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, kNumberTypeComplex128, complex128),
  DTYPE_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, kNumberTypeComplex64, complex64),
  DTYPE_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, double),
  DTYPE_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, float),
  DTYPE_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t),
  DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, int32_t),
  DTYPE_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt16, int16_t),
  DTYPE_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, int8_t),
  DTYPE_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeUInt64, uint64_t),
  DTYPE_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeUInt32, uint32_t),
  DTYPE_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeUInt16, uint16_t),
  DTYPE_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, uint8_t),
  DTYPE_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, kNumberTypeComplex128, complex128),
  DTYPE_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, kNumberTypeComplex64, complex64),
};

std::vector<KernelAttr> ScatterNdCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ScatterNdFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScatterNd, ScatterNdCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
