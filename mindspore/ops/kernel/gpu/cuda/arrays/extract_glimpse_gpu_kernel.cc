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
#include "kernel/gpu/cuda/arrays/extract_glimpse_gpu_kernel.h"
#include <string>
#include <algorithm>
#include "include/curand.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "mindspore/ops/infer/extract_glimpse.h"
namespace mindspore {
namespace kernel {
constexpr int64_t INPUTS_DIMS = 4;
constexpr int64_t SIZE_DIMS = 1;
constexpr int64_t OFFSETS_DIMS = 2;
constexpr size_t kExtractGlimpseInputsNum = 3;
constexpr size_t kExtractGlimpseOutputsNum = 1;
constexpr size_t kExtractGlimpseTwo = 2;
constexpr int64_t kExtractGlimpseOne = 2;
constexpr int64_t kExtractGlimpseThree = 3;
bool ExtractGlimpseGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kExtractGlimpseInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kExtractGlimpseOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  centered_ = GetValue<bool>(primitive_->GetAttr("centered"));
  normalized_ = GetValue<bool>(primitive_->GetAttr("normalized"));
  uniform_noise_ = GetValue<bool>(primitive_->GetAttr("uniform_noise"));
  noise_ = kExtractGlimpsenoiseMap[GetValue<std::string>(primitive_->GetAttr("noise"))];
  return true;
}

int ExtractGlimpseGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  inputs_shape = inputs[kIndex0]->GetShapeVector();
  size_shape = inputs[kIndex1]->GetShapeVector();
  offsets_shape = inputs[kIndex2]->GetShapeVector();
  output_shape = outputs[kIndex0]->GetShapeVector();
  if (offsets_shape[1] != kExtractGlimpseOne) {
    MS_LOG(EXCEPTION) << "The second dimension of offsets must be 2, but got " << offsets_shape[1] << ".";
  }
  if (offsets_shape[0] != inputs_shape[0]) {
    MS_LOG(EXCEPTION) << "The first dimension of offsets must be consistent with the first dimension of x, but got "
                      << offsets_shape[0] << ".";
  }
  batch_cnt_ = inputs_shape[0];
  image_height_ = inputs_shape[1];
  image_width_ = inputs_shape[kExtractGlimpseOne];
  channels_ = inputs_shape[kExtractGlimpseThree];
  if (channels_ == 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the last dimension of x can not be zero.";
  }
  inputs_elements_ = batch_cnt_ * image_height_ * image_width_ * channels_;
  size_elements_ = kExtractGlimpseTwo;
  offsets_elements_ = batch_cnt_ * kExtractGlimpseTwo;
  output_elements_ =
    output_shape[0] * output_shape[1] * output_shape[kExtractGlimpseOne] * output_shape[kExtractGlimpseThree];
  auto GetNums = [](const std::vector<int64_t> &shape) {
    size_t res = 1;
    for (const auto &sh : shape) {
      res *= LongToSize(sh);
    }
    return res;
  };
  inputs_size_ = abstract::TypeIdSize(inputs[kIndex0]->dtype_id()) * GetNums(inputs_shape);
  size_size_ = abstract::TypeIdSize(inputs[kIndex1]->dtype_id()) * GetNums(size_shape);
  offsets_size_ = abstract::TypeIdSize(inputs[kIndex2]->dtype_id()) * GetNums(offsets_shape);
  output_size_ = abstract::TypeIdSize(outputs[kIndex0]->dtype_id()) * output_elements_;
  return KRET_OK;
}

template <typename T>
bool ExtractGlimpseGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &workspace,
                                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *x = GetDeviceAddress<T>(inputs, kIndex0);
  int *size = GetDeviceAddress<int>(inputs, kIndex1);
  T *offsets = GetDeviceAddress<T>(inputs, kIndex2);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  stream_ptr_ = stream_ptr;
  cudaError_t ret = CalExtractGlimpse(output_elements_, batch_cnt_, channels_, image_height_, image_width_, noise_,
                                      centered_, normalized_, uniform_noise_, x, size, offsets, output,
                                      reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(ret, "ExtractGlimpseGpuKernelMod");
  return true;
}
std::vector<std::pair<KernelAttr, ExtractGlimpseGpuKernelMod::ExtractGlimpseFunc>>
  ExtractGlimpseGpuKernelMod::func_list_ = {{KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat32)
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeFloat32)
                                               .AddOutputAttr(kNumberTypeFloat32),
                                             &ExtractGlimpseGpuKernelMod::LaunchKernel<float>}};
std::vector<KernelAttr> ExtractGlimpseGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ExtractGlimpseFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ExtractGlimpse, ExtractGlimpseGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
