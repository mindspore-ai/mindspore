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

#include "kernel/gpu/cuda/nn/adaptive_max_pool3d_grad_gpu_kernel.h"
#include <utility>

namespace mindspore {
namespace kernel {
constexpr int64_t maxIndexIdx = 2;

namespace {
template <typename T, typename S>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateAdaptiveMaxPoolGradKernelPtr(const std::string &kernel_name,
                                                                                  const uint32_t &device_id) {
  return std::make_unique<cukernel::AdaptiveMaxPoolGradHelperGpuKernel<T, S>>(kernel_name, device_id);
}

using AdaptiveMaxPoolGradPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

#define REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(TypeId1, TypeId2, Type1, Type2)                            \
  {                                                                                                        \
    KernelAttr().AddInputAttr(TypeId1).AddInputAttr(TypeId1).AddInputAttr(TypeId2).AddOutputAttr(TypeId1), \
      CreateAdaptiveMaxPoolGradKernelPtr<Type1, Type2>                                                     \
  }

const std::vector<std::pair<KernelAttr, AdaptiveMaxPoolGradPtrCreatorFunc>> kernel_attr = {
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeFloat16, kNumberTypeInt32, half, int32_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeFloat32, kNumberTypeInt32, float, int32_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeFloat64, kNumberTypeInt32, double, int32_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeInt8, kNumberTypeInt32, int8_t, int32_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeInt16, kNumberTypeInt32, int16_t, int32_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeInt32, kNumberTypeInt32, int32_t, int32_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeInt64, kNumberTypeInt32, int64_t, int32_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeUInt8, kNumberTypeInt32, uint8_t, int32_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeUInt16, kNumberTypeInt32, uint16_t, int32_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeUInt32, kNumberTypeInt32, uint32_t, int32_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeUInt64, kNumberTypeInt32, uint64_t, int32_t),

  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeFloat16, kNumberTypeInt64, half, int64_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeInt32, kNumberTypeInt64, int32_t, int64_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeUInt16, kNumberTypeInt64, uint16_t, int64_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeUInt32, kNumberTypeInt64, uint32_t, int64_t),
  REG_ADAPTIVE_MAX_POOL3D_GRAD_GPU_KERNEL(kNumberTypeUInt64, kNumberTypeInt64, uint64_t, int64_t),
};  // namespace
}  // namespace

bool AdaptiveMaxPool3DGradGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &workspace,
                                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemsetAsync(output_ptrs[0], 0, outputs[0]->size(), reinterpret_cast<cudaStream_t>(stream_ptr)),
    "failed to set cuda memory with zeros.");

  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool AdaptiveMaxPool3DGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  helper_ptr_->SetKernelParam(attr_ptr_);

  return true;
}

int AdaptiveMaxPool3DGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> input_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> x_shape = inputs[1]->GetShapeVector();
  std::vector<int64_t> index_shape = inputs[maxIndexIdx]->GetShapeVector();
  std::vector<int64_t> out_shape = outputs[0]->GetShapeVector();

  (void)input_shapes.emplace_back(input_shape);
  (void)input_shapes.emplace_back(x_shape);
  (void)input_shapes.emplace_back(index_shape);
  (void)output_shapes.emplace_back(out_shape);

  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

std::vector<KernelAttr> AdaptiveMaxPool3DGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AdaptiveMaxPoolGradPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AdaptiveMaxPool3DGrad, AdaptiveMaxPool3DGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
