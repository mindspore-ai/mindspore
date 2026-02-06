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

#include "kernel/gpu/cuda/math/cumulative_logsumexp_gpu_kernel.h"
#include "mindspore/ops/infer/cumulative_logsumexp.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCumulativeLogsumexpStaticInputsNum = 1;
constexpr size_t kCumulativeLogsumexpDynamicInputsNum = 2;
}  // namespace

bool CumulativeLogsumexpGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  auto input_num = inputs.size();
  if (input_num == kCumulativeLogsumexpStaticInputsNum) {
    is_dynamic_shape_ = false;
  } else if (input_num == kCumulativeLogsumexpDynamicInputsNum) {
    is_dynamic_shape_ = true;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 2 or 3, but got " << input_num;
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int CumulativeLogsumexpGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto shape = inputs.at(kIndex0)->GetShapeVector();
  shape_.clear();
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_), LongToSize);
  is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
  if (is_null_input_) {
    return true;
  }

  exclusive_ = GetValue<bool>(primitive_->GetAttr("exclusive"));
  reverse_ = GetValue<bool>(primitive_->GetAttr("reverse"));
  if (!is_dynamic_shape_) {
    axis_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("axis")));
    Reshape();
  }
  return KRET_OK;
}

void CumulativeLogsumexpGpuKernelMod::Reshape() {
  axis_ = (axis_ < 0) ? axis_ + SizeToInt(shape_.size()) : axis_;
  dims_[kIndex0] = 1;
  dims_[kIndex1] = shape_[IntToSize(axis_)];
  dims_[kIndex2] = 1;
  for (size_t i = 0; i < IntToSize(axis_); i++) {
    dims_[kIndex0] *= shape_[i];
  }
  for (size_t i = IntToSize(axis_) + 1; i < shape_.size(); i++) {
    dims_[kIndex2] *= shape_[i];
  }
  stride_ = dims_[kIndex1] * dims_[kIndex2];
  stride2_ = dims_[kIndex2];
}

template <typename T>
bool CumulativeLogsumexpGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &workspace,
                                                   const std::vector<KernelTensor *> &outputs) {
  if (is_null_input_) {
    return true;
  }
  auto input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  auto output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  auto any = [](auto... args) -> bool { return ((args == nullptr) || ...); };
  if (any(input_addr, output_addr, cuda_stream_)) {
    return false;
  }
  if (is_dynamic_shape_) {
    const auto &axis_addr = inputs.at(kIndex1);
    MS_EXCEPTION_IF_NULL(axis_addr);
    if (axis_addr->size() == sizeof(int)) {
      int axis_tmp;
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(&axis_tmp, axis_addr->device_ptr(), axis_addr->size(), cudaMemcpyDeviceToHost, cuda_stream_),
        "For '" << kernel_name_ << "', cudaMemcpyAsync input 'axis' device to host failed.");
      if (cudaStreamQuery(cuda_stream_) != cudaSuccess) {
        CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "cuda Stream Sync Failed");
      }
      axis_ = axis_tmp;
    } else if (inputs.at(kIndex1)->size() == sizeof(int64_t)) {
      int64_t axis_tmp;
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(&axis_tmp, axis_addr->device_ptr(), axis_addr->size(), cudaMemcpyDeviceToHost, cuda_stream_),
        "For '" << kernel_name_ << "', cudaMemcpyAsync input 'axis' device to host failed.");
      if (cudaStreamQuery(cuda_stream_) != cudaSuccess) {
        CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "cuda Stream Sync Failed");
      }
      axis_ = static_cast<int>(axis_tmp);
    } else if (inputs.at(kIndex1)->size() == sizeof(int16_t)) {
      int16_t axis_tmp;
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(&axis_tmp, axis_addr->device_ptr(), axis_addr->size(), cudaMemcpyDeviceToHost, cuda_stream_),
        "For '" << kernel_name_ << "', cudaMemcpyAsync input 'axis' device to host failed.");
      if (cudaStreamQuery(cuda_stream_) != cudaSuccess) {
        CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "cuda Stream Sync Failed");
      }
      axis_ = static_cast<int>(axis_tmp);
    } else {
      MS_LOG(ERROR) << "The dtype of 'axis' should be int16, int32 or int64";
      return false;
    }
    auto input_dim_length_ = SizeToInt(shape_.size());
    if (axis_ >= input_dim_length_ || axis_ < -input_dim_length_) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << ", 'axis' should be less than the length of 'input' dimension, but got 'axis': " << axis_
                    << " and the length of 'input' dimension: " << input_dim_length_;
      return false;
    }
    Reshape();
  }
  auto status = CumulativeLogsumexp(input_addr, output_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], stride_,
                                    stride2_, exclusive_, reverse_, device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, CumulativeLogsumexpGpuKernelMod::CumulativeLogsumexpLaunchFunc>>
  CumulativeLogsumexpGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CumulativeLogsumexpGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CumulativeLogsumexpGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CumulativeLogsumexpGpuKernelMod::LaunchKernel<double>},
    // Dynamic shape related.
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &CumulativeLogsumexpGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &CumulativeLogsumexpGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &CumulativeLogsumexpGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
     &CumulativeLogsumexpGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &CumulativeLogsumexpGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &CumulativeLogsumexpGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat16),
     &CumulativeLogsumexpGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),
     &CumulativeLogsumexpGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat64),
     &CumulativeLogsumexpGpuKernelMod::LaunchKernel<double>},
};

std::vector<KernelAttr> CumulativeLogsumexpGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, CumulativeLogsumexpGpuKernelMod::CumulativeLogsumexpLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CumulativeLogsumexp, CumulativeLogsumexpGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
