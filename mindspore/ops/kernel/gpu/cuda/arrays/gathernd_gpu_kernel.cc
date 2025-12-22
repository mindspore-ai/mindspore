/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/gathernd_gpu_kernel.h"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S>
bool GatherNdFwdGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  VARIABLE_NOT_USED(workspace);

  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  S *indices_addr = GetDeviceAddress<S>(inputs, 1);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);

  // strides and indices
  GatherNdInfo<S> info;
  for (int64_t i = 0; i < dim_indices_last_; ++i) {
    info.indices[i] = static_cast<S>(batch_indices_[i]);
    info.strides[i] = static_cast<S>(batch_strides_[i]);
  }

  auto status = GatherNd(input_addr, indices_addr, output_addr, dims_[0], dims_[1], dims_[2], info,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define REG_INDEX_ND(DT1, DT2, T1, T2) \
  {KernelAttr().AddInputAttr(DT1).AddInputAttr(DT2).AddOutputAttr(DT1), &GatherNdFwdGpuKernelMod::LaunchKernel<T1, T2>}

#define GATHER_ND_GPU_REGISTER(DT, T) \
  REG_INDEX_ND(DT, kNumberTypeInt64, T, int64_t), REG_INDEX_ND(DT, kNumberTypeInt32, T, int32_t)

std::vector<std::pair<KernelAttr, GatherNdFwdGpuKernelMod::GatherNdFwdFunc>> GatherNdFwdGpuKernelMod::func_list_ = {
  GATHER_ND_GPU_REGISTER(kNumberTypeComplex64, cuComplex),
  GATHER_ND_GPU_REGISTER(kNumberTypeComplex128, cuDoubleComplex),
  GATHER_ND_GPU_REGISTER(kNumberTypeFloat16, half),
  GATHER_ND_GPU_REGISTER(kNumberTypeFloat32, float),
  GATHER_ND_GPU_REGISTER(kNumberTypeFloat64, double),
  GATHER_ND_GPU_REGISTER(kNumberTypeInt8, char),
  GATHER_ND_GPU_REGISTER(kNumberTypeInt16, int16_t),
  GATHER_ND_GPU_REGISTER(kNumberTypeInt32, int),
  GATHER_ND_GPU_REGISTER(kNumberTypeInt64, int64_t),
  GATHER_ND_GPU_REGISTER(kNumberTypeUInt8, uchar),
  GATHER_ND_GPU_REGISTER(kNumberTypeUInt32, uint),
  GATHER_ND_GPU_REGISTER(kNumberTypeBool, bool),
};

bool GatherNdFwdGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int GatherNdFwdGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  auto input_shapes = inputs[0]->GetShapeVector();
  auto indices_shapes = inputs[1]->GetShapeVector();
  auto &output_shapes = outputs[0]->GetShapeVector();

  is_null_input_ = CHECK_SHAPE_NULL(input_shapes, kernel_name_, "input") ||
                   CHECK_SHAPE_NULL(indices_shapes, kernel_name_, "input_indices") ||
                   CHECK_SHAPE_NULL(output_shapes, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_OK;
  }

  input_shapes_ = input_shapes;
  // make a scalar to tensor whose shape is (1,)
  if (indices_shapes.size() == 0) {
    indices_shapes.emplace_back(1);
  }
  int64_t dim_of_indices = 1;
  for (size_t i = 0; i < indices_shapes.size() - IntToSize(1); i++) {
    dim_of_indices *= indices_shapes[i];
  }

  int64_t dim_after_indices = 1;
  dim_indices_last_ = indices_shapes[indices_shapes.size() - IntToSize(1)];
  for (size_t i = dim_indices_last_; i < input_shapes_.size(); i++) {
    dim_after_indices *= input_shapes_[i];
  }
  dims_ = {LongToSize(dim_of_indices), LongToSize(dim_after_indices), LongToSize(dim_indices_last_)};

  batch_strides_.resize(dim_indices_last_, 0);
  batch_indices_.resize(dim_indices_last_, 0);

  if (dim_indices_last_ > 0) {
    batch_strides_[dim_indices_last_ - 1] = input_shapes_[dim_indices_last_ - 1];
    batch_indices_[dim_indices_last_ - 1] = dims_[1];
  }
  for (int i = static_cast<int>(dim_indices_last_) - 1; i > 0; --i) {
    batch_strides_[i - 1] = input_shapes_[i - 1];
    batch_indices_[i - 1] = batch_indices_[i] * input_shapes_[i];
  }

  return ret;
}

std::vector<KernelAttr> GatherNdFwdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GatherNdFwdFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, GatherNd, GatherNdFwdGpuKernelMod);

}  // namespace kernel
}  // namespace mindspore
