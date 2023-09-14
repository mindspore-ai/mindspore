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

#include "plugin/device/gpu/kernel/math/geqrf_gpu_kernel.h"
#include <complex>
#include <map>
#include <utility>
#include <vector>
#include "abstract/utils.h"
#include "include/common/utils/convert_utils.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/real_to_complex_impl.cuh"

namespace mindspore {
namespace kernel {
bool GeqrfGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Geqrf>(base_operator);
  kernel_name_ = kernel_ptr->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in [ "
                  << "float32, float64, complex64, complex128], but got: " << kernel_attr << ".";
    return false;
  }
  launch_kernel_func_ = func_list_[index].second.first;
  init_lists_func_ = func_list_[index].second.second;
  handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
  return true;
}

int GeqrfGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  input_x_shape_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                       inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  input_x_dims_ = input_x_shape_.size();
  is_null_input_ = (input_x_dims_ == 0);
  if (is_null_input_) {
    init_lists_func_(this);
    return 0;
  }
  if (input_x_dims_ < kDim2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', dimensions must be greater than or equal to 2"
                  << ", but got [" << input_x_dims_ << "].";
    return KRET_RESIZE_FAILED;
  }
  m_ = input_x_shape_[input_x_dims_ - kDim2];
  n_ = input_x_shape_[input_x_dims_ - kDim1];
  p_ = std::min(m_, n_);
  batch_size_ = 1;
  if (input_x_dims_ >= kDim3) {
    for (size_t i = 0; i < input_x_dims_ - kDim2; i++) {
      batch_size_ = batch_size_ * input_x_shape_[i];
    }
  }
  for (size_t i = 0; i < input_x_dims_; ++i) {
    transpose_input_x_shape_[i] = input_x_shape_[i];
    if (i == input_x_dims_ - kDim2) {
      transpose_input_x_axis_[i] = input_x_dims_ - kDim1;
      transpose_output_y_shape_[i] = input_x_shape_[input_x_dims_ - kDim1];
    } else if (i == input_x_dims_ - kDim1) {
      transpose_input_x_axis_[i] = input_x_dims_ - kDim2;
      transpose_output_y_shape_[i] = input_x_shape_[input_x_dims_ - kDim2];
    } else {
      transpose_input_x_axis_[i] = i;
      transpose_output_y_shape_[i] = input_x_shape_[i];
    }
  }

  init_lists_func_(this);
  return KRET_OK;
}

template <typename T>
void GeqrfGpuKernelMod::InitSizeLists() {
  // input x
  input_size_list_.push_back(batch_size_ * m_ * n_ * sizeof(T));
  // output y, tau
  output_size_list_.push_back(batch_size_ * m_ * n_ * sizeof(T));
  output_size_list_.push_back(batch_size_ * p_ * sizeof(T));
  workspace_size_list_.push_back(batch_size_ * sizeof(int));
  // for transpose input x and output y, tau
  workspace_size_list_.push_back(batch_size_ * m_ * n_ * sizeof(T));
  workspace_size_list_.push_back(batch_size_ * m_ * n_ * sizeof(T));
}

template <typename T>
void GeqrfGpuKernelMod::RunGeqrf(const size_t m, const size_t n, T *d_a, int *dev_info, T *d_output_y, T *output_tau) {
  int lwork = 0;
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnSgeqrf_bufferSize(handle_, m, n, d_a, m, &lwork),
                                           "cusolver query orgqr work size failed.");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnDgeqrf_bufferSize(handle_, m, n, d_a, m, &lwork),
                                           "cusolver query orgqr work size failed.");
  } else if constexpr (std::is_same_v<T, Complex<float>>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnCgeqrf_bufferSize(handle_, m, n, reinterpret_cast<cuComplex *>(d_a), m, &lwork),
      "cusolver query orgqr work size failed.");
  } else if constexpr (std::is_same_v<T, Complex<double>>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnZgeqrf_bufferSize(handle_, m, n, reinterpret_cast<cuDoubleComplex *>(d_a), m, &lwork),
      "cusolver query orgqr work size failed.");
  }

  void *d_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(T) * lwork);
  if (d_work == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the memory of d_work alloc failed.";
  }
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnSgeqrf(handle_, m, n, d_a, m, output_tau, static_cast<T *>(d_work), lwork, dev_info),
      "cusolver orgqr failed.");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnDgeqrf(handle_, m, n, d_a, m, output_tau, static_cast<T *>(d_work), lwork, dev_info),
      "cusolver orgqr failed.");
  } else if constexpr (std::is_same_v<T, Complex<float>>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnCgeqrf(handle_, m, n, reinterpret_cast<cuComplex *>(d_a), m, reinterpret_cast<cuComplex *>(output_tau),
                       reinterpret_cast<cuComplex *>(d_work), lwork, dev_info),
      "cusolver orgqr failed.");
  } else if constexpr (std::is_same_v<T, Complex<double>>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnZgeqrf(handle_, m, n, reinterpret_cast<cuDoubleComplex *>(d_a), m,
                       reinterpret_cast<cuDoubleComplex *>(output_tau), reinterpret_cast<cuDoubleComplex *>(d_work),
                       lwork, dev_info),
      "cusolver orgqr failed.");
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(d_output_y, d_a, sizeof(T) * m_ * n_, cudaMemcpyDeviceToDevice,
                                                     reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "cuda memcpy output A failed!");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "In Geqrf kernel, cuda Stream Sync Failed.");
  }
  device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_work);
}

void GeqrfGpuKernelMod::CheckResult(int *dev_info) {
  std::vector<int> info_gpu(batch_size_, 0);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(info_gpu.data(), dev_info, sizeof(int) * batch_size_, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "Copy device result failed");
  for (size_t i = 0; i < info_gpu.size(); ++i) {
    if (info_gpu[i] != 0) {
      MS_LOG(INFO) << "For '" << kernel_name_ << "', the compute result has wrong value. The " << -info_gpu[i]
                   << "th parameter is wrong (not counting handle) in batch " << i << " data.";
    }
  }
}

template <typename T>
void GeqrfGpuKernelMod::LaunchGeqrf(T *d_input_x, T *d_output_y, T *output_tau, int *dev_info) {
  for (size_t batch = 0; batch < batch_size_; ++batch) {
    RunGeqrf(m_, n_, d_input_x + batch * m_ * n_, dev_info + batch, d_output_y + batch * m_ * n_,
             output_tau + batch * p_);
  }

  CheckResult(dev_info);
}

template <typename T>
bool GeqrfGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(handle_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                "CusolverDnSetStream failed");

  T *input_x = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_y = GetDeviceAddress<T>(outputs, kIndex0);
  T *output_tau = GetDeviceAddress<T>(outputs, kIndex1);

  int *dev_info = GetDeviceAddress<int>(workspace, kIndex0);
  T *d_input_x = GetDeviceAddress<T>(workspace, kIndex1);
  T *d_output_y = GetDeviceAddress<T>(workspace, kIndex2);

  TransposeInfo x_info;
  TransposeInfo y_info;
  for (size_t i = 0; i < input_x_dims_; ++i) {
    x_info.input_shape.push_back(static_cast<int64_t>(transpose_input_x_shape_[i]));
    x_info.perm.push_back(static_cast<int32_t>(transpose_input_x_axis_[i]));
    y_info.input_shape.push_back(static_cast<int64_t>(transpose_output_y_shape_[i]));
    y_info.perm.push_back(static_cast<int32_t>(transpose_input_x_axis_[i]));
  }

  auto s1 = CalTranspose<T, true>(batch_size_ * m_ * n_, input_x, x_info, d_input_x,
                                  reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(s1, "Transpose called by " + kernel_name_);
  LaunchGeqrf(d_input_x, d_output_y, output_tau, dev_info);
  auto s2 = CalTranspose<T, true>(batch_size_ * m_ * n_, d_output_y, y_info, output_y,
                                  reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(s2, "Transpose called by " + kernel_name_);

  return true;
}

std::vector<std::pair<KernelAttr, std::pair<GeqrfGpuKernelMod::LaunchKernelFunc, GeqrfGpuKernelMod::InitSizeListsFunc>>>
  GeqrfGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     {&GeqrfGpuKernelMod::LaunchKernel<float>, &GeqrfGpuKernelMod::InitSizeLists<float>}},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     {&GeqrfGpuKernelMod::LaunchKernel<double>, &GeqrfGpuKernelMod::InitSizeLists<double>}},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     {&GeqrfGpuKernelMod::LaunchKernel<Complex<float>>, &GeqrfGpuKernelMod::InitSizeLists<Complex<float>>}},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     {&GeqrfGpuKernelMod::LaunchKernel<Complex<double>>, &GeqrfGpuKernelMod::InitSizeLists<Complex<double>>}},
};

std::vector<KernelAttr> GeqrfGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, std::pair<LaunchKernelFunc, InitSizeListsFunc>> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Geqrf, GeqrfGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
