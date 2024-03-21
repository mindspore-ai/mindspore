/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include <string>
#include <algorithm>
#include <functional>

#include "plugin/device/gpu/kernel/arrays/gatherd_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S>
bool GatherDGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  VARIABLE_NOT_USED(workspace);

  auto input_addr = reinterpret_cast<T *>(inputs.at(kIndex0)->device_ptr());
  auto index_addr = reinterpret_cast<S *>(inputs.at(kIndex2)->device_ptr());
  auto output_addr = reinterpret_cast<T *>(outputs.at(kIndex0)->device_ptr());
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto status = GatherD(input_addr, index_addr, output_addr, dim_, index_num_, rank_, input_shape_helper_,
                        index_shape_helper_, cuda_stream, GET_CTX_DEVICE_ID);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define REG_INDEX(DT1, DT2, T1, T2)                      \
  {                                                      \
    KernelAttr()                                         \
      .AddInputAttr(DT1)                                 \
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
      .AddInputAttr(DT2)                                 \
      .AddOutputAttr(DT1),                               \
      &GatherDGpuKernelMod::LaunchKernel<T1, T2>         \
  }

#define GATHER_D_GPU_REGISTER(DT, T) \
  REG_INDEX(DT, kNumberTypeInt64, T, int64_t), REG_INDEX(DT, kNumberTypeInt32, T, int32_t)

std::vector<std::pair<KernelAttr, GatherDGpuKernelMod::GatherFwdFunc>> GatherDGpuKernelMod::func_list_ = {
  GATHER_D_GPU_REGISTER(kNumberTypeComplex64, Complex<float>),
  GATHER_D_GPU_REGISTER(kNumberTypeComplex128, Complex<double>),
  GATHER_D_GPU_REGISTER(kNumberTypeFloat16, half),
  GATHER_D_GPU_REGISTER(kNumberTypeFloat32, float),
  GATHER_D_GPU_REGISTER(kNumberTypeFloat64, double),
  GATHER_D_GPU_REGISTER(kNumberTypeInt8, uchar),
  GATHER_D_GPU_REGISTER(kNumberTypeInt16, int16_t),
  GATHER_D_GPU_REGISTER(kNumberTypeInt32, int32_t),
  GATHER_D_GPU_REGISTER(kNumberTypeInt64, int64_t),
  GATHER_D_GPU_REGISTER(kNumberTypeUInt8, uint8_t),
  GATHER_D_GPU_REGISTER(kNumberTypeUInt16, uint16_t),
  GATHER_D_GPU_REGISTER(kNumberTypeUInt32, uint32_t),
  GATHER_D_GPU_REGISTER(kNumberTypeUInt64, uint64_t),
  GATHER_D_GPU_REGISTER(kNumberTypeBool, bool)};

bool GatherDGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int GatherDGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  auto input_shapes = inputs[0]->GetShapeVector();
  auto index_shapes = inputs[kIndex2]->GetShapeVector();
  auto &output_shapes = outputs[0]->GetShapeVector();

  is_null_input_ = CHECK_SHAPE_NULL(input_shapes, kernel_name_, "input") ||
                   CHECK_SHAPE_NULL(index_shapes, kernel_name_, "input_indices") ||
                   CHECK_SHAPE_NULL(output_shapes, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_OK;
  }

  if (input_shapes.empty()) {
    input_shapes = ShapeVector{1};
  }

  if (index_shapes.empty()) {
    index_shapes = ShapeVector{1};
  }

  auto dim = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  if (dim < 0) {
    dim += index_shapes.size();
  }
  dim_ = static_cast<size_t>(dim);

  rank_ = input_shapes.size();
  index_num_ =
    static_cast<size_t>(std::accumulate(index_shapes.begin(), index_shapes.end(), 1, std::multiplies<int64_t>()));

  if (input_shapes.size() > kMaxShapeRank) {
    MS_LOG(INTERNAL_EXCEPTION) << "The kernel GatherD only support Tensor whose rank is less than " << kMaxShapeRank
                               << ", but got rank: " << input_shapes.size() << ".";
  }

  for (size_t i = 0; i < input_shapes.size(); i++) {
    input_shape_helper_.shape[i] = static_cast<size_t>(input_shapes[i]);
    index_shape_helper_.shape[i] = static_cast<size_t>(index_shapes[i]);
  }

  return KRET_OK;
}

std::vector<KernelAttr> GatherDGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GatherFwdFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, GatherD, GatherDGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
