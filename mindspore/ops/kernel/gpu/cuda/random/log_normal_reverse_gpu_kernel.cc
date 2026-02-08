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

#include "kernel/gpu/cuda/random/log_normal_reverse_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const uint32_t kNumInput = 1;
const uint32_t kNumOutput = 1;
}  // namespace

bool LogNormalReverseGpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                          const std::vector<kernel::KernelTensor *> &workspace,
                                          const std::vector<kernel::KernelTensor *> &outputs, void *stream_ptr) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kNumInput, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kNumOutput, kernel_name_);
  stream_ptr_ = stream_ptr;
  CHECK_CURAND_RET_WITH_EXCEPT(curandSetStream(mask_generator_, reinterpret_cast<cudaStream_t>(stream_ptr_)),
                               "Failed to set stream for generator");
  MS_EXCEPTION_IF_NULL(mask_generator_);

  return kernel_func_(this, inputs, workspace, outputs);
}

bool LogNormalReverseGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the inputs and outputs should not be empty, but got empty. ";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  if (inputs.size() != kNumInput || outputs.size() != kNumOutput) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kNumInput << " and "
                  << kNumOutput << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be float32, "
                  << "but got: " << kernel_attr << ".";
    return false;
  }

  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
  input_dtype_ = inputs.at(kIndex0)->dtype_id();
  output_dtype_ = outputs.at(kIndex0)->dtype_id();
  if (input_dtype_ != kNumberTypeFloat32 && input_dtype_ != kNumberTypeFloat16 && input_dtype_ != kNumberTypeFloat64) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the datatype of the input should be in "
                      << "[ float16, float32, float64 ], "
                      << "but got: " << input_dtype_ << ".";
  }
  if (input_dtype_ != output_dtype_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", the data type of the input should match the data type of the output, "
                      << "but got input type: " << input_dtype_ << " and output type: " << output_dtype_ << ".";
  }

  input_mean_ = GetValue<float>(primitive_->GetAttr("mean"));
  input_std_ = GetValue<float>(primitive_->GetAttr("std"));

  kernel_func_ = func_list_[pair.second].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);

  if (!states_init_) {
    int64_t seed = time(NULL);
    seed_ = static_cast<uint64_t>(seed);
    CHECK_CURAND_RET_WITH_EXCEPT(curandCreateGenerator(&mask_generator_, CURAND_RNG_PSEUDO_PHILOX4_32_10),
                                 "Failed to create generator");
    CHECK_CURAND_RET_WITH_EXCEPT(curandSetPseudoRandomGeneratorSeed(mask_generator_, seed_),
                                 "Failed to SetPseudoRandomGeneratorSeed");
    MS_EXCEPTION_IF_NULL(mask_generator_);
    states_init_ = true;
  }

  return true;
}

int LogNormalReverseGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> input_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                          inputs.at(kIndex0)->GetDeviceShapeVector().end());
  input_elements_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<int64_t>());
  size_t input_size = input_elements_ * unit_size_;
  output_size_list_.push_back(input_size);
  workspace_size_list_.push_back(input_size);
  return KRET_OK;
}

void LogNormalReverseGpuKernelMod::ResetResource() noexcept {
  input_elements_ = 0;
  output_size_list_.clear();
  workspace_size_list_.clear();
}

bool LogNormalReverseGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &workspace,
                                                const std::vector<kernel::KernelTensor *> &outputs) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr_);

  if (output_dtype_ == kNumberTypeFloat32) {
    float *output = GetDeviceAddress<float>(outputs, kIndex0);
    size_t elem_num = inputs[0]->size() / sizeof(float);

    CHECK_CURAND_RET_WITH_EXCEPT(curandGenerateLogNormal(mask_generator_, output, elem_num, input_mean_, input_std_),
                                 "Failed to generate lognormal");
  } else if (output_dtype_ == kNumberTypeFloat64) {
    double *output = GetDeviceAddress<double>(outputs, kIndex0);
    size_t elem_num = inputs[0]->size() / sizeof(double);

    CHECK_CURAND_RET_WITH_EXCEPT(
      curandGenerateLogNormalDouble(mask_generator_, output, elem_num, input_mean_, input_std_),
      "Failed to generate lognormal");
  } else if (output_dtype_ == kNumberTypeFloat16) {
    half *input = GetDeviceAddress<half>(inputs, kIndex0);
    half *output = GetDeviceAddress<half>(outputs, kIndex0);
    float *mask_h = GetDeviceAddress<float>(workspace, kDim0);
    size_t elem_num = inputs[0]->size() / sizeof(half);

    CHECK_CURAND_RET_WITH_EXCEPT(curandGenerateLogNormal(mask_generator_, mask_h, elem_num, input_mean_, input_std_),
                                 "Failed to generate lognormal");
    auto status = CalLogNormalReverseHalf(input, output, elem_num, mask_h, cuda_stream_);
    CHECK_CUDA_STATUS(status, kernel_name_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, LogNormalReverseGpuKernelMod::LogNormalReverseFunc>>
  LogNormalReverseGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &LogNormalReverseGpuKernelMod::LaunchKernel},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &LogNormalReverseGpuKernelMod::LaunchKernel},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &LogNormalReverseGpuKernelMod::LaunchKernel}};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LogNormalReverse, LogNormalReverseGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
