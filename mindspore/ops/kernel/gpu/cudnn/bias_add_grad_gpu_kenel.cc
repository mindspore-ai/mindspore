/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cudnn/bias_add_grad_gpu_kenel.h"
#include "kernel/gpu/kernel_constants.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/bias_add_grad_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

bool BiasAddGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  InitResource();
  return true;
}

int BiasAddGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  auto dy_shape = inputs[kIndex0]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(dy_shape, kernel_name_, "input");
  if (is_null_input_ || IsDynamic(dy_shape)) {
    return KRET_UNKNOWN_SHAPE;
  }
  ResetResource();
  auto dtype = inputs[kIndex0]->dtype_id();
  unit_size_ = abstract::TypeIdSize(dtype);

  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(dtype));
  num_dims_ = dy_shape.size();
  auto input_device_format = inputs[kIndex0]->format();
  cudnn_compute_format_ = (input_device_format == Format::NHWC) ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
  data_format_ = input_device_format;
  auto format = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  size_t pos = 1;
  if (format == Format::NHWC) {
    data_format_ = format;
    pos = dy_shape.size() - 1;
  }
  bias_size_ = LongToSizeClipNeg(dy_shape[pos]);
  constexpr size_t four_4D = 4;
  size_t num_dims_fix = std::max(num_dims_, four_4D);
  for (size_t i = 0; i < num_dims_fix; i++) {
    dy_shape_.push_back((i < num_dims_) ? dy_shape[i] : 1);
    db_shape_.push_back((i == pos) ? dy_shape[i] : 1);
    if (dy_shape_[i] != db_shape_[i]) {
      same_dims_ = false;
    }
  }
  dy_num_ *= SizeOf(dy_shape_);
  db_num_ *= SizeOf(db_shape_);
  MethodSelection();
  SetResource();
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool BiasAddGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs) {
  T *dy_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *db_addr = GetDeviceAddress<T>(outputs, kIndex0);
  if (same_dims_) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(db_addr, dy_addr, output_size_list_[kIndex0], cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_)),
      "cudaMemcpyAsync failed.");
  } else {
    if (use_cudnn_) {  // shared memory not satisfied or num_dim > 4
      T *indices_addr = GetPossiblyNullDeviceAddress<T>(workspace, kIndex0);
      T *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, kIndex1);
      const float alpha = 1;
      const float beta = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnReduceTensor(cudnn_handle_, op_desc_, indices_addr, workspace_size_list_[kIndex0], workspace_addr,
                          workspace_size_list_[kIndex1], &alpha, dy_desc_, dy_addr, &beta, db_desc_, db_addr),
        "cudnnReduceTensor failed");
    } else {  // use own implementation which is more efficient but cannot process num_dim > 4
      cudaError_t status = cudaErrorNotReady;
      if (data_format_ == Format::NHWC) {
        status = CalBiasAddGradNHWC(dy_num_, bias_size_, dy_addr, db_addr, reinterpret_cast<cudaStream_t>(stream_));
        CHECK_CUDA_STATUS(status, kernel_name_);
      } else {
        status = CalBiasAddGradNCHW(dy_num_, bias_size_, SizeToInt(dy_shape_[kIndex2]), SizeToInt(dy_shape_[kIndex3]),
                                    dy_addr, db_addr, reinterpret_cast<cudaStream_t>(stream_));
        CHECK_CUDA_STATUS(status, kernel_name_);
      }
    }
  }
  return true;
}

void BiasAddGradGpuKernelMod::DestroyResource() noexcept {
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyReduceTensorDescriptor(op_desc_),
                                      "cudnnDestroyReduceTensorDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(db_desc_), "cudnnDestroyTensorDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(dy_desc_), "cudnnDestroyOpTensorDescriptor failed");
}

void BiasAddGradGpuKernelMod::ResetResource() noexcept {
  same_dims_ = true;
  is_null_input_ = false;
  use_cudnn_ = false;
  dy_num_ = 1;
  db_num_ = 1;
  num_dims_ = 0;
  bias_size_ = 0;
  dy_shape_.clear();
  db_shape_.clear();
  data_format_ = Format::NCHW;
  cudnn_data_type_ = CUDNN_DATA_FLOAT;
  cudnn_compute_format_ = CUDNN_TENSOR_NCHW;
  output_size_list_.clear();
  workspace_size_list_.clear();
}

void BiasAddGradGpuKernelMod::MethodSelection() {
  // opt implementation can only process num_dims_ <= 4
  // for num_dims_ = 2, not time-consuming, use cudnn
  if (num_dims_ > kDim4 || num_dims_ == kDim2) {
    use_cudnn_ = true;
    return;
  }
  if (data_format_ == Format::NHWC) {
    constexpr auto tile_size_large_mat = 32;
    auto required_shared_mem_size = tile_size_large_mat * (tile_size_large_mat + 1) * unit_size_;
    // nhwc opt implementation performs not so well when bias_size_ <= 6
    constexpr auto max_cudnn_bias_size = 6;
    if (required_shared_mem_size > SHARED_MEM_PER_BLOCK || bias_size_ <= max_cudnn_bias_size) {
      use_cudnn_ = true;
      return;
    }
  }
}

void BiasAddGradGpuKernelMod::InitResource() {
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dy_desc_), "cudnnCreateTensorDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&db_desc_), "cudnnCreateTensorDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateReduceTensorDescriptor(&op_desc_),
                                      "cudnnCreateOpTensorDescriptor failed");
}

void BiasAddGradGpuKernelMod::SetResource() {
  if (use_cudnn_) {
    // Expand to 4 dims for cudnnSetTensorNdDescriptorEx.
    constexpr size_t four_4D = 4;
    size_t cudnn_dims = std::max(num_dims_, four_4D);
    std::unique_ptr<int[]> dy_dims = std::make_unique<int[]>(cudnn_dims);
    std::unique_ptr<int[]> db_dims = std::make_unique<int[]>(cudnn_dims);
    for (size_t i = 0; i < cudnn_dims; i++) {
      dy_dims[i] = LongToInt(dy_shape_[i]);
      db_dims[i] = LongToInt(db_shape_[i]);
    }
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetTensorNdDescriptorEx(dy_desc_, cudnn_compute_format_, cudnn_data_type_,
                                                                     SizeToInt(cudnn_dims), dy_dims.get()),
                                        "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetTensorNdDescriptorEx(db_desc_, cudnn_compute_format_, cudnn_data_type_,
                                                                     SizeToInt(cudnn_dims), db_dims.get()),
                                        "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetReduceTensorDescriptor(op_desc_, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN,
                                     CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES),
      "cudnnSetReduceTensorDescriptor failed");
  }
}

void BiasAddGradGpuKernelMod::InitSizeLists() {
  if (use_cudnn_) {
    size_t dy_size, db_size;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(dy_desc_, &dy_size),
                                        "cudnnGetTensorSizeInBytes failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(db_desc_, &db_size),
                                        "cudnnGetTensorSizeInBytes failed");
    output_size_list_.push_back(db_size);
    size_t indices_size, workspace_size;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetReductionIndicesSize(cudnn_handle_, op_desc_, dy_desc_, db_desc_, &indices_size),
      "cudnnGetReductionIndicesSize failed")
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetReductionWorkspaceSize(cudnn_handle_, op_desc_, dy_desc_, db_desc_, &workspace_size),
      "cudnnGetReductionWorkspaceSize failed")
    workspace_size_list_.push_back(indices_size);
    workspace_size_list_.push_back(workspace_size);
  } else {
    output_size_list_.push_back(db_num_ * unit_size_);
  }
}

const std::vector<std::pair<KernelAttr, BiasAddGradGpuKernelMod::KernelRunFunc>> &BiasAddGradGpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, BiasAddGradGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &BiasAddGradGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &BiasAddGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     &BiasAddGradGpuKernelMod::LaunchKernel<int8_t>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BiasAddGrad, BiasAddGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
