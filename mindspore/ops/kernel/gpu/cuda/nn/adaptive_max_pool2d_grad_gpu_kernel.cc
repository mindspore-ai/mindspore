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

#include "kernel/gpu/cuda/nn/adaptive_max_pool2d_grad_gpu_kernel.h"
#include <utility>

namespace mindspore {
namespace kernel {
constexpr int64_t maxIndexIdx = 2;

namespace {
template <typename T, typename S>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateAdaptiveMaxPool2DGradKernelPtr(const std::string &kernel_name,
                                                                                    const uint32_t &device_id) {
  return std::make_unique<cukernel::AdaptiveMaxPoolGradHelperGpuKernel<T, S>>(kernel_name, device_id);
}

using AdaptiveMaxPool2DGradPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, AdaptiveMaxPool2DGradPtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   CreateAdaptiveMaxPool2DGradKernelPtr<half, int64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   CreateAdaptiveMaxPool2DGradKernelPtr<float, int64_t>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   CreateAdaptiveMaxPool2DGradKernelPtr<double, int64_t>},
};
}  // namespace

bool AdaptiveMaxPool2DGradGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
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

bool AdaptiveMaxPool2DGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
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

int AdaptiveMaxPool2DGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
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

std::vector<KernelAttr> AdaptiveMaxPool2DGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, AdaptiveMaxPool2DGradPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AdaptiveMaxPool2DGrad, AdaptiveMaxPool2DGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
