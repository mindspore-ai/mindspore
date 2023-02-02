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

#include "plugin/device/gpu/kernel/math/qr_gpu_kernel.h"
#include <type_traits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_public/cusolver.h"

namespace mindspore {
namespace kernel {
template <typename R>
using Complex = mindspore::utils::Complex<R>;

constexpr size_t kNum2 = 2;
bool QrGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                          const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::Qr>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in [float32, float64, complex64, "
                  << "complex128], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_input_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  full_matrices_ = kernel_ptr_->get_full_matrices();
  cusolverH_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
  return true;
}

int QrGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                           const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  ResetResource();
  std::vector<int64_t> output_shape = outputs.at(kIndex0)->GetShapeVector();
  size_t output_elements = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  if (output_elements == 0) {
    is_null_input_ = true;
  }

  std::vector<size_t> x_shape = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                    inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  total_size_ = std::accumulate(x_shape.begin(), x_shape.end(), size_t(1), std::multiplies<size_t>());
  input_dims_ = x_shape.size();
  if (input_dims_ < kDim2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', dimensions must be greater than or equal to 2"
                  << ", but got [" << input_dims_ << "].";
    return KRET_RESIZE_FAILED;
  }
  m_ = x_shape[input_dims_ - kDim2];
  n_ = x_shape[input_dims_ - kDim1];
  if (full_matrices_) {
    p_ = m_;
  } else {
    p_ = std::min(m_, n_);
  }
  s_ = std::max(m_, n_);

  batch_size_ = 1;
  for (size_t i = 0; i < input_dims_ - kDim2; i++) {
    batch_size_ = batch_size_ * x_shape[i];
  }

  // transpose row and col
  for (size_t i = 0; i < input_dims_; ++i) {
    transpose_input_shape_[i] = x_shape[i];
    if (i == input_dims_ - kDim2) {
      transpose_input_axis_[i] = input_dims_ - kDim1;
    } else if (i == input_dims_ - kDim1) {
      transpose_input_axis_[i] = input_dims_ - kDim2;
    } else {
      transpose_input_axis_[i] = i;
    }
  }

  input_size_list_ = {total_size_ * unit_input_size_};
  output_size_list_ = {batch_size_ * m_ * p_ * unit_input_size_, batch_size_ * p_ * n_ * unit_input_size_};
  workspace_size_list_ = {batch_size_ * sizeof(int),
                          input_dims_ * sizeof(size_t),
                          input_dims_ * sizeof(size_t),
                          total_size_ * unit_input_size_,
                          batch_size_ * m_ * p_ * unit_input_size_,
                          batch_size_ * m_ * n_ * unit_input_size_,
                          batch_size_ * n_ * unit_input_size_,
                          batch_size_ * m_ * s_ * unit_input_size_,
                          batch_size_ * m_ * n_ * unit_input_size_,
                          kNum2 * sizeof(size_t),
                          kNum2 * sizeof(size_t)};

  return 0;
}

template <typename T>
void QrGpuKernelMod::RunQr(T *d_input, T *d_A, T *d_tau, int *dev_info, T *d_output_q, T *d_output_r) {
  const size_t lda = m_;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(d_A, d_input, sizeof(T) * m_ * n_, cudaMemcpyDeviceToDevice, stream),
    "copy device A result to host failed");
  int geqrf_work_size = 0;
  cusolver::geqrf_buffersize<T>(cusolverH_, m_, n_, d_A, lda, &geqrf_work_size);
  int orgqr_work_size = 0;
  cusolver::orgqr_buffersize<T>(cusolverH_, m_, p_, p_, d_A, lda, d_tau, &orgqr_work_size);
  int lwork = geqrf_work_size > orgqr_work_size ? geqrf_work_size : orgqr_work_size;

  void *d_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(T) * lwork);
  if (d_work == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the memory of d_work alloc failed.";
  }
  // compute QR factorization
  cusolver::geqrf<T>(cusolverH_, m_, n_, d_A, lda, d_tau, static_cast<T *>(d_work), lwork, dev_info);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(d_output_r, d_A, sizeof(T) * m_ * n_, cudaMemcpyDeviceToDevice, stream),
    "Copy to QR factorization device result failed");
  // compute Q=H(1)*H(2)*...*H(K)
  cusolver::orgqr<T>(cusolverH_, m_, p_, p_, d_A, lda, d_tau, static_cast<T *>(d_work), lwork, dev_info);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(d_output_q, d_A, sizeof(T) * m_ * p_, cudaMemcpyDeviceToDevice, stream),
    "copy device Q result to host failed");
  device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_work);
}

template <typename T>
void QrGpuKernelMod::LaunchQr(T *d_input, T *d_A, T *d_tau, T *d_output_q, T *d_output_r, int *dev_info,
                              size_t *d_transpose_shape, size_t *d_transpose_axis, T *d_output_r_t, T *output_r) {
  for (size_t batch = 0; batch < batch_size_; ++batch) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
    RunQr(d_input + batch * m_ * n_, d_A + batch * m_ * s_, d_tau + batch * n_, dev_info + batch,
          d_output_q + batch * m_ * p_, d_output_r + batch * m_ * n_);
    CalTranspose(m_ * n_, d_output_r + batch * m_ * n_, d_transpose_shape, d_transpose_axis, kNum2,
                 d_output_r_t + batch * m_ * n_, stream);
    CalTriu(p_ * n_, d_output_r_t + batch * m_ * n_, 0, p_, n_, output_r + batch * p_ * n_, device_id_, stream);
  }
}

template <typename T>
bool QrGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &workspace,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(cusolverH_, stream), "CusolverDnSetStream failed");
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_q = GetDeviceAddress<T>(outputs, kIndex0);
  T *output_r = GetDeviceAddress<T>(outputs, kIndex1);

  int *dev_info = GetDeviceAddress<int>(workspace, kIndex0);
  size_t *d_transpose_input_shape = GetDeviceAddress<size_t>(workspace, kIndex1);
  size_t *d_transpose_input_axis = GetDeviceAddress<size_t>(workspace, kIndex2);
  T *d_input = GetDeviceAddress<T>(workspace, kIndex3);
  T *d_output_q = GetDeviceAddress<T>(workspace, kIndex4);
  T *d_output_r = GetDeviceAddress<T>(workspace, kIndex5);
  T *d_tau = GetDeviceAddress<T>(workspace, kIndex6);
  T *d_A = GetDeviceAddress<T>(workspace, kIndex7);
  T *d_output_r_t = GetDeviceAddress<T>(workspace, kIndex8);
  size_t *d_transpose_shape = GetDeviceAddress<size_t>(workspace, kIndex9);
  size_t *d_transpose_axis = GetDeviceAddress<size_t>(workspace, kIndex10);

  size_t transpose_shape[2] = {n_, m_};
  size_t transpose_axis[2] = {1, 0};
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(d_transpose_input_axis, transpose_input_axis_,
                                                     sizeof(size_t) * input_dims_, cudaMemcpyHostToDevice, stream),
                                     "For Qr transpose_input_axis cuda memcpy failed!");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(d_transpose_input_shape, transpose_input_shape_,
                                                     sizeof(size_t) * input_dims_, cudaMemcpyHostToDevice, stream),
                                     "For Qr transpose_input_shape cuda memcpy failed!");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(d_transpose_shape, transpose_shape, sizeof(size_t) * kNum2, cudaMemcpyHostToDevice, stream),
    "For Qr transpose_shape cuda memcpy failed!");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(d_transpose_axis, transpose_axis, sizeof(size_t) * kNum2, cudaMemcpyHostToDevice, stream),
    "For Qr transpose_axis cuda memcpy failed!");

  CalTranspose(total_size_, input, d_transpose_input_shape, d_transpose_input_axis, input_dims_, d_input, stream);
  LaunchQr(d_input, d_A, d_tau, d_output_q, d_output_r, dev_info, d_transpose_shape, d_transpose_axis, d_output_r_t,
           output_r);

  for (size_t i = 0; i < input_dims_; i++) {
    transpose_q_shape_[i] = transpose_input_shape_[i];
  }
  transpose_q_shape_[input_dims_ - kDim2] = p_;
  transpose_q_shape_[input_dims_ - kDim1] = m_;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(d_transpose_input_shape, transpose_q_shape_,
                                                     sizeof(size_t) * input_dims_, cudaMemcpyHostToDevice, stream),
                                     "cuda memcpy failed!");
  CalTranspose(batch_size_ * m_ * p_, d_output_q, d_transpose_input_shape, d_transpose_input_axis, input_dims_,
               output_q, stream);
  return true;
}

std::vector<std::pair<KernelAttr, QrGpuKernelMod::LaunchKernelFunc>> QrGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &QrGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &QrGpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64),
   &QrGpuKernelMod::LaunchKernel<Complex<float>>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128),
   &QrGpuKernelMod::LaunchKernel<Complex<double>>}};

std::vector<KernelAttr> QrGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LaunchKernelFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Qr, QrGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
