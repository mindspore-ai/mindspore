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

#include "plugin/device/gpu/kernel/math/orgqr_gpu_kernel.h"
#include <complex>
#include <vector>
#include <map>
#include <utility>
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "include/common/utils/convert_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/real_to_complex_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_public/cusolver.h"

namespace mindspore {
namespace kernel {
bool OrgqrGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Orgqr>(base_operator);
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

int OrgqrGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
  input_tau_shape_ = std::vector<size_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                         inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  input_x_dims_ = input_x_shape_.size();
  input_tau_dims_ = input_tau_shape_.size();
  is_null_input_ = (input_x_dims_ == 0 || input_tau_dims_ == 0);
  if (is_null_input_) {
    init_lists_func_(this);
    return 0;
  }
  if (input_x_dims_ < kDim2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', dimensions must be greater than or equal to 2"
                  << ", but got [" << input_x_dims_ << "].";
    return KRET_RESIZE_FAILED;
  }
  if (input_x_dims_ - kDim1 != input_tau_dims_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', tau should have one dimension less than x"
                  << ", but rank of x is" << input_x_dims_ << " and "
                  << "rank of tau is " << input_tau_dims_ << ".";
    return KRET_RESIZE_FAILED;
  }
  m_ = input_x_shape_[input_x_dims_ - kDim2];
  n_ = input_x_shape_[input_x_dims_ - kDim1];
  k_ = input_tau_shape_[input_tau_dims_ - kDim1];
  if (m_ < n_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', x.shape[-2] must be greater than "
                  << "or equal to x.shape[-1], but x.shape[-2] is " << m_ << " and "
                  << "x.shape[-1] is " << n_ << ".";
    return KRET_RESIZE_FAILED;
  }
  if (n_ < k_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', x.shape[-1] must be greater than "
                  << "or equal to tau.shape[-1], but x.shape[-1] is " << n_ << " and "
                  << "tau.shape[-1] is " << k_ << ".";
    return KRET_RESIZE_FAILED;
  }
  batch_size_ = 1;
  if (input_x_dims_ >= kDim3) {
    for (size_t i = 0; i < input_x_dims_ - kDim2; i++) {
      if (input_x_shape_[i] != input_tau_shape_[i]) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', x and tau should share the same batch size, but x.shape[" << i
                      << "] is " << input_x_shape_[i] << ", and tau.shape[" << i << "] is " << input_tau_shape_[i];
        return KRET_RESIZE_FAILED;
      }
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
void OrgqrGpuKernelMod::InitSizeLists() {
  // input x, tau
  input_size_list_.push_back(batch_size_ * m_ * n_ * sizeof(T));
  input_size_list_.push_back(batch_size_ * k_ * sizeof(T));
  // output y
  output_size_list_.push_back(batch_size_ * m_ * n_ * sizeof(T));
  workspace_size_list_.push_back(batch_size_ * sizeof(int));
  // for transpose input x and output y
  workspace_size_list_.push_back(input_x_dims_ * sizeof(size_t));
  workspace_size_list_.push_back(input_x_dims_ * sizeof(size_t));
  workspace_size_list_.push_back(batch_size_ * m_ * n_ * sizeof(T));
  workspace_size_list_.push_back(batch_size_ * m_ * n_ * sizeof(T));
  workspace_size_list_.push_back(input_x_dims_ * sizeof(size_t));
}

template <typename T>
void OrgqrGpuKernelMod::RunOrgqr(const size_t m, const size_t n, const size_t k, T *d_a, T *tau, int *dev_info,
                                 T *output_y) {
  int lwork = 0;
  cusolver::orgqr_buffersize<T>(handle_, m, n, k, d_a, m, tau, &lwork);

  void *d_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(T) * lwork);
  if (d_work == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the memory of d_work alloc failed.";
  }
  cusolver::orgqr<T>(handle_, m, n, k, d_a, m, tau, static_cast<T *>(d_work), lwork, dev_info);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(output_y, d_a, sizeof(T) * m_ * n_, cudaMemcpyDeviceToDevice,
                                                     reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "cuda memcpy output A failed!");
  device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_work);
}

void OrgqrGpuKernelMod::CheckResult(int *dev_info) {
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
void OrgqrGpuKernelMod::LaunchOrgqr(T *d_input_x, T *input_tau, T *d_output_y, int *dev_info) {
  for (size_t batch = 0; batch < batch_size_; ++batch) {
    RunOrgqr(m_, n_, k_, d_input_x + batch * m_ * n_, input_tau + batch * k_, dev_info + batch,
             d_output_y + batch * m_ * n_);
  }

  CheckResult(dev_info);
}

template <typename T>
bool OrgqrGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(handle_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                "CusolverDnSetStream failed");

  T *input_x = GetDeviceAddress<T>(inputs, kIndex0);
  T *input_tau = GetDeviceAddress<T>(inputs, kIndex1);
  T *output_y = GetDeviceAddress<T>(outputs, kIndex0);

  int *dev_info = GetDeviceAddress<int>(workspace, kIndex0);
  size_t *d_trans_input_x_shape = GetDeviceAddress<size_t>(workspace, kIndex1);
  size_t *d_trans_input_x_axis = GetDeviceAddress<size_t>(workspace, kIndex2);
  T *d_input_x = GetDeviceAddress<T>(workspace, kIndex3);
  T *d_output_y = GetDeviceAddress<T>(workspace, kIndex4);
  size_t *d_trans_output_y_shape = GetDeviceAddress<size_t>(workspace, kIndex5);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(d_trans_input_x_axis, transpose_input_x_axis_, sizeof(size_t) * input_x_dims_,
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "cuda memcpy failed!");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(d_trans_input_x_shape, transpose_input_x_shape_, sizeof(size_t) * input_x_dims_,
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "cuda memcpy failed!");
  CalTranspose(batch_size_ * m_ * n_, input_x, d_trans_input_x_shape, d_trans_input_x_axis, input_x_dims_, d_input_x,
               reinterpret_cast<cudaStream_t>(cuda_stream_));

  LaunchOrgqr(d_input_x, input_tau, d_output_y, dev_info);
  cudaMemcpyAsync(d_trans_output_y_shape, transpose_output_y_shape_, sizeof(size_t) * input_x_dims_,
                  cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CalTranspose(batch_size_ * m_ * n_, d_output_y, d_trans_output_y_shape, d_trans_input_x_axis, input_x_dims_, output_y,
               reinterpret_cast<cudaStream_t>(cuda_stream_));

  return true;
}

std::vector<std::pair<KernelAttr, std::pair<OrgqrGpuKernelMod::LaunchKernelFunc, OrgqrGpuKernelMod::InitSizeListsFunc>>>
  OrgqrGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     {&OrgqrGpuKernelMod::LaunchKernel<float>, &OrgqrGpuKernelMod::InitSizeLists<float>}},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     {&OrgqrGpuKernelMod::LaunchKernel<double>, &OrgqrGpuKernelMod::InitSizeLists<double>}},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     {&OrgqrGpuKernelMod::LaunchKernel<Complex<float>>, &OrgqrGpuKernelMod::InitSizeLists<Complex<float>>}},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     {&OrgqrGpuKernelMod::LaunchKernel<Complex<double>>, &OrgqrGpuKernelMod::InitSizeLists<Complex<double>>}},
};

std::vector<KernelAttr> OrgqrGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, std::pair<LaunchKernelFunc, InitSizeListsFunc>> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Orgqr, OrgqrGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
