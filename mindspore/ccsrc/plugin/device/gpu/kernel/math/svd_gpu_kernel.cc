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

#include "plugin/device/gpu/kernel/math/svd_gpu_kernel.h"

#include <map>
#include <utility>

namespace mindspore {
namespace kernel {
bool SvdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                           const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Svd>(base_operator);
  kernel_name_ = kernel_ptr->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  launch_kernel_func_ = func_list_[index].second.first;
  init_size_lists_func_ = func_list_[index].second.second;
  compute_uv_ = kernel_ptr->compute_uv();
  full_matrices_ = kernel_ptr->full_matrices();
  job_ = compute_uv_ ? (full_matrices_ ? 'A' : 'S') : 'N';
  handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
  return true;
}

int SvdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs,
                            const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  if (IsDynamicRank(input_shape)) {
    return KRET_OK;
  }
  DestroyResource();
  ResetResource();
  input_shape_ = Convert2SizeTClipNeg(input_shape);
  total_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (total_size_ == 0);
  if (is_null_input_) {
    init_size_lists_func_(this);
    return KRET_OK;
  }
  dims_ = input_shape_.size();
  if (dims_ < kDim2) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dimensions must >= 2, but got [" << dims_;
  }

  m_ = input_shape_[dims_ - kDim2];
  n_ = input_shape_[dims_ - kDim1];
  p_ = std::min(m_, n_);
  if (m_ >= n_) {
    m_ge_n_ = true;
  }
  batch_size_ = 1;
  for (size_t i = 0; i < dims_ - kDim2; i++) {
    batch_size_ = batch_size_ * input_shape_[i];
  }
  for (size_t i = 0; i < dims_; ++i) {
    transpose_input_shape_[i] = input_shape_[i];
    if (i == dims_ - kDim2) {
      transpose_input_axis_[i] = dims_ - kDim1;
    } else if (i == dims_ - kDim1) {
      transpose_input_axis_[i] = dims_ - kDim2;
    } else {
      transpose_input_axis_[i] = i;
    }
  }
  init_size_lists_func_(this);
  constexpr auto kBatchedMaxRowCol = 32;
  if (m_ <= kBatchedMaxRowCol && n_ <= kBatchedMaxRowCol && batch_size_ > 1 && (full_matrices_ || m_ == n_)) {
    batched_ = true;
  }
  return 0;
}

template <typename T>
void SvdGpuKernelMod::InitSizeLists() {
  // input a
  input_size_list_.push_back(total_size_ * sizeof(T));
  // output s, u, v
  output_size_list_.push_back(batch_size_ * p_ * sizeof(T));
  if (compute_uv_) {
    if (full_matrices_) {
      output_size_list_.push_back(batch_size_ * m_ * m_ * sizeof(T));
      output_size_list_.push_back(batch_size_ * n_ * n_ * sizeof(T));
    } else {
      output_size_list_.push_back(batch_size_ * m_ * p_ * sizeof(T));
      output_size_list_.push_back(batch_size_ * n_ * p_ * sizeof(T));
    }
  } else {
    output_size_list_.push_back(0);
    output_size_list_.push_back(0);
  }
  workspace_size_list_.push_back(batch_size_ * sizeof(int));  // for dev_info

  // for transpose input
  workspace_size_list_.push_back(dims_ * sizeof(size_t));
  workspace_size_list_.push_back(dims_ * sizeof(size_t));
  workspace_size_list_.push_back(total_size_ * sizeof(T));

  // for dev_u and dev_v
  if (compute_uv_) {
    if (full_matrices_) {
      workspace_size_list_.push_back(batch_size_ * m_ * m_ * sizeof(T));
      workspace_size_list_.push_back(batch_size_ * n_ * n_ * sizeof(T));
    } else {
      workspace_size_list_.push_back(batch_size_ * m_ * p_ * sizeof(T));
      workspace_size_list_.push_back(batch_size_ * n_ * p_ * sizeof(T));
    }
  }
}

template <typename T>
void SvdGpuKernelMod::RunSvd(const size_t m, const size_t n, const size_t batch, T *d_a, int *dev_info, T *output_s,
                             T *d_output_u, T *d_output_v) {
  int lwork = 0;

  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnSgesvd_bufferSize(handle_, m, n, &lwork),
                                           "cusolver query svd work size fail");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnDgesvd_bufferSize(handle_, m, n, &lwork),
                                           "cusolver query svd work size fail");
  }

  void *d_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(T) * lwork);
  if (d_work == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the memory of d_work alloc failed.";
  }
  T *d_s1 = output_s + batch * p_;
  T *d_u1 = nullptr;
  T *d_v1 = nullptr;
  if (compute_uv_) {
    if (full_matrices_) {
      d_u1 = d_output_u + batch * m * m;
      d_v1 = d_output_v + batch * n * n;
    } else {
      d_u1 = d_output_u + batch * m * p_;
      d_v1 = d_output_v + batch * n * p_;
    }
  }

  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnSgesvd(handle_, job_, job_, m, n, d_a, m, d_s1, d_u1, m, d_v1, n,
                                                            static_cast<T *>(d_work), lwork, nullptr, dev_info),
                                           "cusolver svd fail");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnDgesvd(handle_, job_, job_, m, n, d_a, m, d_s1, d_u1, m, d_v1, n,
                                                            static_cast<T *>(d_work), lwork, nullptr, dev_info),
                                           "cusolver svd fail");
  }

  device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_work);
}

template <typename T>
void SvdGpuKernelMod::RunSvdBatched(const size_t m, const size_t n, T *d_input, T *output_s, T *output_u, T *output_v,
                                    int *dev_info) {
  cusolverEigMode_t jobz = compute_uv_ ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  int lwork = 0;
  gesvdjInfo_t info;
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnCreateGesvdjInfo(&info), "cusolver svd fail");
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnSgesvdjBatched_bufferSize(handle_, jobz, m, n, d_input, m, output_s, output_u, m, output_v, n, &lwork,
                                          info, batch_size_),
      "cusolver query svd work size fail");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnDgesvdjBatched_bufferSize(handle_, jobz, m, n, d_input, m, output_s, output_u, m, output_v, n, &lwork,
                                          info, batch_size_),
      "cusolver query svd work size fail");
  }

  void *work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(T) * lwork);
  if (work == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the memory of work alloc failed.";
  }

  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnSgesvdjBatched(handle_, jobz, m, n, d_input, m, output_s, output_u, m, output_v, n,
                               static_cast<T *>(work), lwork, dev_info, info, batch_size_),
      "cusolver svd fail");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnDgesvdjBatched(handle_, jobz, m, n, d_input, m, output_s, output_u, m, output_v, n,
                               static_cast<T *>(work), lwork, dev_info, info, batch_size_),
      "cusolver svd fail");
  }

  device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(work);
}

template <typename T>
void SvdGpuKernelMod::TransposeUV(const size_t m, const size_t n, size_t *d_transpose_input_shape,
                                  size_t *d_transpose_input_axis, T *d_output_u, T *d_output_v, T *output_u,
                                  T *output_v) {
  if (full_matrices_) {
    transpose_input_shape_[dims_ - kDim2] = m;
    transpose_input_shape_[dims_ - kDim1] = m;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(d_transpose_input_shape, transpose_input_shape_, sizeof(size_t) * dims_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "cuda memcpy failed!");
    CalTranspose(batch_size_ * m * m, d_output_u, d_transpose_input_shape, d_transpose_input_axis, dims_, output_u,
                 reinterpret_cast<cudaStream_t>(cuda_stream_));
    if (batched_) {
      transpose_input_shape_[dims_ - kDim2] = n;
      transpose_input_shape_[dims_ - kDim1] = n;
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(d_transpose_input_shape, transpose_input_shape_, sizeof(size_t) * dims_, cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(cuda_stream_)),
        "cuda memcpy failed!");
      CalTranspose(batch_size_ * n * n, d_output_v, d_transpose_input_shape, d_transpose_input_axis, dims_, output_v,
                   reinterpret_cast<cudaStream_t>(cuda_stream_));
    } else {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(output_v, d_output_v, sizeof(size_t) * batch_size_ * n * n, cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(cuda_stream_)),
        "cuda memcpy failed!");
    }
  } else {
    transpose_input_shape_[dims_ - kDim2] = p_;
    transpose_input_shape_[dims_ - kDim1] = m;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(d_transpose_input_shape, transpose_input_shape_, sizeof(size_t) * dims_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "cuda memcpy failed!");
    CalTranspose(batch_size_ * m * p_, d_output_u, d_transpose_input_shape, d_transpose_input_axis, dims_, output_u,
                 reinterpret_cast<cudaStream_t>(cuda_stream_));

    if (batched_) {
      transpose_input_shape_[dims_ - kDim2] = p_;
      transpose_input_shape_[dims_ - kDim1] = n;
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(d_transpose_input_shape, transpose_input_shape_, sizeof(size_t) * dims_, cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(cuda_stream_)),
        "cuda memcpy failed!");
      CalTranspose(batch_size_ * n * p_, d_output_v, d_transpose_input_shape, d_transpose_input_axis, dims_, output_v,
                   reinterpret_cast<cudaStream_t>(cuda_stream_));
    } else {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(output_v, d_output_v, sizeof(size_t) * batch_size_ * n * p_, cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(cuda_stream_)),
        "cuda memcpy failed!");
    }
  }
}

void SvdGpuKernelMod::CheckResult(int *dev_info) {
  std::vector<int> info_gpu(batch_size_, 0);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(info_gpu.data(), dev_info, sizeof(int) * batch_size_, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "Copy to device result failed");
  for (size_t i = 0; i < info_gpu.size(); ++i) {
    if (info_gpu[i] < 0) {
      MS_LOG(INFO) << "For '" << kernel_name_ << "', the compute result has wrong value. The " << -info_gpu[i]
                   << "th parameter is wrong (not counting handle) in batch " << i << " data.";
    } else if (info_gpu[i] > 0) {
      MS_LOG(INFO) << "For '" << kernel_name_ << "', in batch " << i << " data, there are " << info_gpu[i]
                   << " superdiagonals of an intermediate bidiagonal from did not coverage to zero";
    }
  }
}

template <typename T>
void SvdGpuKernelMod::LaunchSvd(const size_t m, const size_t n, T *d_input, T *output_s, T *output_u, T *output_v,
                                T *d_output_u, T *d_output_v, int *dev_info, size_t *d_transpose_input_shape,
                                size_t *d_transpose_input_axis) {
  if (batched_) {
    RunSvdBatched(m, n, d_input, output_s, d_output_u, d_output_v, dev_info);
  } else {
    for (size_t batch = 0; batch < batch_size_; ++batch) {
      RunSvd(m, n, batch, d_input + batch * m_ * n_, dev_info + batch, output_s, d_output_u, d_output_v);
    }
  }

  if (compute_uv_) {
    TransposeUV(m, n, d_transpose_input_shape, d_transpose_input_axis, d_output_u, d_output_v, output_u, output_v);
  }

  CheckResult(dev_info);
}

template <typename T>
bool SvdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(handle_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                "CusolverDnSetStream failed");

  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_s = GetDeviceAddress<T>(outputs, kIndex0);
  T *output_u = nullptr;
  T *output_v = nullptr;
  T *d_output_u = nullptr;
  T *d_output_v = nullptr;
  if (compute_uv_) {
    output_u = GetDeviceAddress<T>(outputs, kIndex1);
    output_v = GetDeviceAddress<T>(outputs, kIndex2);
    // Store output u and v before transpose.
    d_output_u = GetDeviceAddress<T>(workspace, kIndex4);
    d_output_v = GetDeviceAddress<T>(workspace, kIndex5);
  }

  int *dev_info = GetDeviceAddress<int>(workspace, kIndex0);

  size_t *d_transpose_input_shape = GetDeviceAddress<size_t>(workspace, kIndex1);
  size_t *d_transpose_input_axis = GetDeviceAddress<size_t>(workspace, kIndex2);
  T *d_input = GetDeviceAddress<T>(workspace, kIndex3);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(d_transpose_input_axis, transpose_input_axis_, sizeof(size_t) * dims_, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "cuda memcpy failed!");

  if (m_ge_n_) {
    // Because cudaSovler expects column-major matrix, we need transpose A.
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(d_transpose_input_shape, transpose_input_shape_, sizeof(size_t) * dims_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "cuda memcpy failed!");
    CalTranspose(total_size_, input, d_transpose_input_shape, d_transpose_input_axis, dims_, d_input,
                 reinterpret_cast<cudaStream_t>(cuda_stream_));
    LaunchSvd(m_, n_, d_input, output_s, output_u, output_v, d_output_u, d_output_v, dev_info, d_transpose_input_shape,
              d_transpose_input_axis);
  } else {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(d_input, input, sizeof(T) * total_size_, cudaMemcpyHostToDevice,
                                                       reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "cuda memcpy failed!");
    LaunchSvd(n_, m_, d_input, output_s, output_v, output_u, d_output_v, d_output_u, dev_info, d_transpose_input_shape,
              d_transpose_input_axis);
  }

  return true;
}

std::vector<std::pair<KernelAttr, std::pair<SvdGpuKernelMod::LaunchKernelFunc, SvdGpuKernelMod::InitSizeListsFunc>>>
  SvdGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     {&SvdGpuKernelMod::LaunchKernel<float>, &SvdGpuKernelMod::InitSizeLists<float>}},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     {&SvdGpuKernelMod::LaunchKernel<double>, &SvdGpuKernelMod::InitSizeLists<double>}},
};

std::vector<KernelAttr> SvdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, std::pair<LaunchKernelFunc, InitSizeListsFunc>> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Svd, SvdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
