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

#include "plugin/device/gpu/kernel/nn/clip_by_norm_gpu_kernel.h"
#include <memory>
#include <algorithm>
#include <functional>
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cast_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/l2normalize_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/clip_by_norm_impl.cuh"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  ClipByNorm,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ClipByNormGpuKernelMod, float, float)
MS_REG_GPU_KERNEL_TWO(
  ClipByNorm,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
  ClipByNormGpuKernelMod, float, half)
MS_REG_GPU_KERNEL_TWO(
  ClipByNorm,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ClipByNormGpuKernelMod, half, float)
MS_REG_GPU_KERNEL_TWO(
  ClipByNorm,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
  ClipByNormGpuKernelMod, half, half)

namespace {
const std::vector<KernelAttr> supported_kernel_attrs = {
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32)};

// Check whether input, output and workspace address numbers are valid
void CheckAddrNum(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                  const std::vector<AddressPtr> &outputs) {
  constexpr size_t input_addr_num_expected = 2;
  constexpr size_t workspace_addr_num_expected = 6;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == input_addr_num_expected, "The size of input address should be 2.");
  MS_EXCEPTION_IF_CHECK_FAIL(workspace.size() == workspace_addr_num_expected,
                             "The size of workspace address should be 6.");
  MS_EXCEPTION_IF_CHECK_FAIL(outputs.size() == 1, "The size of output address should be 1.");
}
}  // namespace

template <typename T, typename S>
bool ClipByNormGpuKernelMod<T, S>::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  // Get `ClipByNorm` c++ primitive
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = std::dynamic_pointer_cast<ops::ClipByNorm>(base_operator);
  MS_EXCEPTION_IF_NULL(prim);
  kernel_name_ = prim->name();
  // Check whether current input and output attributes are valid.
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  if (!MatchKernelAttr(kernel_attr, GetOpSupport()).first) {
    MS_LOG(ERROR) << "For `" << kernel_name_ << "`, its input or output attributes are not supported.";
    return false;
  }
  return true;
}

template <typename T, typename S>
int ClipByNormGpuKernelMod<T, S>::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = std::dynamic_pointer_cast<ops::ClipByNorm>(base_operator);
  ResetResource();
  InitIOShape(inputs, outputs);
  InitResource();
  CheckTensorSize({x_shape_, clip_norm_shape_, output_shape_});
  InitAxisAndEpsilon(prim);
  BroadcastInfer();
  // Determine data shape, type and format for `inputA_descriptor` and `outputC_descriptor`
  DetermineDeviceDataInfoForCudnn(inputs[0]);
  ChoseCudnnReduceTensorOp();
  InitSizeLists();
  return KRET_OK;
}

template <typename T, typename S>
bool ClipByNormGpuKernelMod<T, S>::Launch(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  CheckAddrNum(inputs, workspace, outputs);
  DoLaunch(inputs, workspace, outputs, stream_ptr);
  return true;
}

template <typename T, typename S>
bool ClipByNormGpuKernelMod<T, S>::DoLaunch(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  // Get address
  T *x_addr = GetDeviceAddress<T>(inputs, kIndex0);
  S *clip_norm_addr = GetDeviceAddress<S>(inputs, kIndex1);
  float *x_float_addr = GetPossiblyNullDeviceAddress<float>(workspace, kIndex0);
  float *l2norm_output_addr = GetPossiblyNullDeviceAddress<float>(workspace, kIndex1);
  float *l2norm_workspace_addr = GetPossiblyNullDeviceAddress<float>(workspace, kIndex2);
  float *div_output_addr = GetPossiblyNullDeviceAddress<float>(workspace, kIndex3);
  float *clip_norm_float_addr = GetPossiblyNullDeviceAddress<float>(workspace, kIndex4);
  float *clip_norm_mul_output_addr = GetPossiblyNullDeviceAddress<float>(workspace, kIndex5);
  float *output_addr = GetDeviceAddress<float>(outputs, 0);
  // Running `cast(x)` to float32 data type
  Cast(x_size_ / sizeof(T), x_addr, x_float_addr, reinterpret_cast<cudaStream_t>(stream_ptr), GET_CTX_DEVICE_ID);
  // Launch `cudnnReduceTensorNorm2` operator to achieve `L2_norm` calculation, keep_dims = true.
  if (all_match_) {
    AbsOp(x_size_ / sizeof(T), x_float_addr, l2norm_output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
  } else {
    constexpr float alpha = 1.0;
    constexpr float beta = 0.0;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, l2norm_workspace_addr,
                        l2_norm_workspace_size_, &alpha, inputA_descriptor_, x_float_addr, &beta, outputC_descriptor_,
                        l2norm_output_addr),
      kernel_name_ + " running cudnnReduceTensor::cudnnReduceTensorNorm2 failed.");
  }
  auto l2_norm_lhs_shape_size = Convert2SizeTClipNeg(l2_norm_lhs_shape_);
  auto l2_norm_rhs_shap_size = Convert2SizeTClipNeg(l2_norm_rhs_shape_);
  auto l2_norm_ouths_shape_size = Convert2SizeTClipNeg(l2_norm_ouths_shape_);
  // Calculation std::max(l2_norm, epsilon) to keep numerical stability.
  GetMaxWithEpsAndValue(l2_norm_output_size_ / sizeof(float), epsilon_, l2norm_output_addr,
                        reinterpret_cast<cudaStream_t>(stream_ptr));
  // Running `x/l2_norm(x)` and broadcast output shape to `input_x` shape
  BroadcastArith(l2_norm_lhs_shape_size, l2_norm_rhs_shap_size, l2_norm_ouths_shape_size, BinaryOpType::kRealDiv,
                 x_float_addr, l2norm_output_addr, div_output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
  // Running `cast(clip_norm)` to the data type of `input_x`
  Cast(clip_norm_size_ / sizeof(S), clip_norm_addr, clip_norm_float_addr, reinterpret_cast<cudaStream_t>(stream_ptr),
       GET_CTX_DEVICE_ID);
  // Running '(x/l2_norm(x)) * clip_norm' and broadcast output shape to `input_x` shape
  if (clip_norm_need_broadcast_) {
    BroadcastArith(l2_norm_ouths_shape_size, Convert2SizeTClipNeg(clip_norm_rhs_shape_), l2_norm_ouths_shape_size,
                   BinaryOpType::kMul, div_output_addr, clip_norm_float_addr, clip_norm_mul_output_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
  } else {
    ElewiseArith(output_size_ / sizeof(float), BinaryOpType::kMul, div_output_addr, clip_norm_float_addr,
                 clip_norm_mul_output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  // Running compare between `input_x` and `upper output` and cast final output to float type.
  CompOp(output_size_ / sizeof(float), x_float_addr, clip_norm_mul_output_addr, output_addr,
         reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

template <typename T, typename S>
std::vector<KernelAttr> ClipByNormGpuKernelMod<T, S>::GetOpSupport() {
  return supported_kernel_attrs;
}

template <typename T, typename S>
void ClipByNormGpuKernelMod<T, S>::DestroyResource() noexcept {
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor_),
                                     kernel_name_ + " running cudnnDestroyReduceTensorDescriptor failed.");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(inputA_descriptor_),
                                     kernel_name_ + " running cudnnDestroyTensorDescriptor failed.");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(outputC_descriptor_),
                                     kernel_name_ + " running cudnnDestroyTensorDescriptor failed.");
}

template <typename T, typename S>
void ClipByNormGpuKernelMod<T, S>::InitResource() {
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor_),
                                      kernel_name_ + " running cudnnCreateReduceTensorDescriptor failed.");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&inputA_descriptor_),
                                      kernel_name_ + " running cudnnCreateTensorDescriptor failed.");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&outputC_descriptor_),
                                      kernel_name_ + " running cudnnCreateTensorDescriptor failed.");
}

template <typename T, typename S>
void ClipByNormGpuKernelMod<T, S>::InitIOShape(const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num_expected = 2;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == input_num_expected, "The size of input tensors should be 2.");
  MS_EXCEPTION_IF_CHECK_FAIL(outputs.size() == 1, "The size of output tensors should be 1.");
  // Get input `x` shape
  MS_EXCEPTION_IF_NULL(inputs[0]);
  x_shape_ = inputs[0]->GetShapeVector();
  if (!IsValidShape(x_shape_)) {
    MS_EXCEPTION(ValueError) << "For " << kernel_name_ << ", input `x` is not supported dynamic shape.";
  }
  x_dim_ = x_shape_.size();
  // Get input `clip_norm` shape
  MS_EXCEPTION_IF_NULL(inputs[1]);
  clip_norm_shape_ = inputs[1]->GetShapeVector();
  if (!IsValidShape(clip_norm_shape_)) {
    MS_EXCEPTION(ValueError) << "For " << kernel_name_ << ", input `clip_norm` is not supported dynamic shape.";
  }
  // Get output shape
  MS_EXCEPTION_IF_NULL(outputs[0]);
  output_shape_ = outputs[0]->GetShapeVector();
  if (!IsValidShape(output_shape_)) {
    MS_EXCEPTION(ValueError) << "For " << kernel_name_ << ", output shape is not supported dynamic shape.";
  }
  MS_EXCEPTION_IF_CHECK_FAIL(output_shape_ == x_shape_, "Output shape should be same with input x shape.");
}

template <typename T, typename S>
void ClipByNormGpuKernelMod<T, S>::InitAxisAndEpsilon(const ops::ClipByNormPtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  // Get axis vector from attribute
  auto axis_value = prim->GetAttr(kAttrAxis);
  MS_EXCEPTION_IF_NULL(axis_value);
  std::vector<int64_t> temp_axis;
  if (axis_value->isa<api::ValueSequence>()) {
    temp_axis = api::GetValue<std::vector<int64_t>>(axis_value);
  } else if (axis_value->isa<api::Int64Imm>()) {
    temp_axis.emplace_back(api::GetValue<int64_t>(axis_value));
  } else {
    MS_EXCEPTION(TypeError) << "For `" << kernel_name_ << "`, the type of attribute `axis` is invalid.";
  }
  // Init `axis_`
  axis_.clear();
  if (temp_axis.empty()) {
    for (size_t i = 0; i < x_dim_; ++i) {
      axis_.emplace_back(i);  // Reduce for all dimensions.
    }
  } else {  // Convert negative `axis` to positive `axis` and keep number unique
    int64_t temp_x_dim = SizeToLong(x_dim_);
    std::for_each(temp_axis.begin(), temp_axis.end(), [this, &temp_x_dim](const int64_t &value) {
      value < 0 ? axis_.emplace_back(LongToSize(value + temp_x_dim)) : axis_.emplace_back(LongToSize(value));
    });
    std::sort(axis_.begin(), axis_.end());
    axis_.erase(std::unique(axis_.begin(), axis_.end()), axis_.end());
  }
}

template <typename T, typename S>
void ClipByNormGpuKernelMod<T, S>::BroadcastInfer() {
  constexpr size_t max_broadcast_dim = 7;
  if (x_dim_ > max_broadcast_dim) {
    MS_EXCEPTION(ValueError) << "For `" << kernel_name_ << "`, the dimension of input args cannot be greater than "
                             << max_broadcast_dim << ", but got `x` dimension: " << x_dim_;
  }
  // Only support `keep_dims=true`
  l2_norm_output_shape_ = output_shape_;
  std::for_each(axis_.begin(), axis_.end(), [this](const size_t &idx) { l2_norm_output_shape_[idx] = 1; });
  // Broadcast infer for 'l2_norm'
  l2_norm_lhs_shape_.resize(max_broadcast_dim, 1);
  l2_norm_rhs_shape_.resize(max_broadcast_dim, 1);
  l2_norm_ouths_shape_.resize(max_broadcast_dim, 1);
  for (size_t i = 0; i < output_shape_.size(); ++i) {
    l2_norm_lhs_shape_[i] = x_shape_[i];
    l2_norm_rhs_shape_[i] = l2_norm_output_shape_[i];
    l2_norm_ouths_shape_[i] = output_shape_[i];
    if (l2_norm_lhs_shape_[i] != l2_norm_rhs_shape_[i]) {
      all_match_ = false;
    }
  }
  // Broadcast infer for '(x/l2_norm(x)) * clip_norm'
  const auto clip_norm_dim = clip_norm_shape_.size();
  clip_norm_need_broadcast_ = clip_norm_shape_ != x_shape_;
  if (clip_norm_need_broadcast_) {
    clip_norm_rhs_shape_.resize(max_broadcast_dim, 1);
    size_t idx = 0;
    for (size_t i = x_dim_ - clip_norm_dim; i < x_dim_; ++i) {
      clip_norm_rhs_shape_[i] = clip_norm_shape_[idx];
      ++idx;
    }
  }
}

template <typename T, typename S>
void ClipByNormGpuKernelMod<T, S>::InitSizeLists() {
  // Init input size list
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(inputA_descriptor_, &x_size_),
                                      kernel_name_ + " running cudnnGetTensorSizeInBytes failed.");
  input_size_list_.emplace_back(x_size_);
  size_t clip_norm_data_type_size = sizeof(S);
  clip_norm_size_ = std::accumulate(clip_norm_shape_.begin(), clip_norm_shape_.end(), clip_norm_data_type_size,
                                    std::multiplies<size_t>());
  clip_norm_size_ = std::max(clip_norm_data_type_size, clip_norm_size_);
  input_size_list_.emplace_back(clip_norm_size_);
  // Init workspace size list
  // size for casting x to float32
  size_t float_type_size = sizeof(float);
  size_t x_float_size = (x_size_ / sizeof(T)) * float_type_size;
  workspace_size_list_.emplace_back(x_float_size);
  // size for saving output of l2_norm(x)
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(outputC_descriptor_, &l2_norm_output_size_),
                                      kernel_name_ + " running cudnnGetTensorSizeInBytes failed.");
  workspace_size_list_.emplace_back(l2_norm_output_size_);
  // size for saving workspace of l2_norm(x)
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetReductionWorkspaceSize(cudnn_handle_, reduce_tensor_descriptor_, inputA_descriptor_, outputC_descriptor_,
                                   &l2_norm_workspace_size_),
    kernel_name_ + " running cudnnGetReductionWorkspaceSize failed.");
  workspace_size_list_.emplace_back(l2_norm_workspace_size_);
  // size for running 'x/l2_norm(x)'
  workspace_size_list_.emplace_back(x_float_size);
  // size for casting clip_norm to float32
  workspace_size_list_.emplace_back((clip_norm_size_ / clip_norm_data_type_size) * float_type_size);
  // size for running '(x/l2_norm(x)) * clip_norm'
  workspace_size_list_.emplace_back(x_float_size);
  // Init output size
  output_size_ = float_type_size * SizeOf(output_shape_);
  output_size_list_.emplace_back(output_size_);
}

template <typename T, typename S>
void ClipByNormGpuKernelMod<T, S>::DetermineDeviceDataInfoForCudnn(const KernelTensorPtr &x_tensor) {
  MS_EXCEPTION_IF_NULL(x_tensor);
  data_type_ = CUDNN_DATA_FLOAT;
  // Determine device data info for `inputA_descriptor`
  constexpr size_t split_dim = 4;
  if (x_dim_ <= split_dim) {
    ShapeVector x_4d_shape;
    ShapeNdTo4d(x_shape_, &x_4d_shape);
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(inputA_descriptor_, CUDNN_TENSOR_NCHW, data_type_, x_4d_shape[kIndex0],
                                 x_4d_shape[kIndex1], x_4d_shape[kIndex2], x_4d_shape[kIndex3]),
      kernel_name_ + " running cudnnSetTensor4dDescriptor failed.");
  } else {
    CudnnSetTensorNdDescriptor(x_shape_, inputA_descriptor_, data_type_, kernel_name_);
  }
  // Determine device data info for `outputC_descriptor`
  if (l2_norm_output_shape_.size() <= split_dim) {
    ShapeVector l2_norm_4d_shape;
    ShapeNdTo4d(l2_norm_output_shape_, &l2_norm_4d_shape);
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(outputC_descriptor_, CUDNN_TENSOR_NCHW, data_type_, l2_norm_4d_shape[kIndex0],
                                 l2_norm_4d_shape[kIndex1], l2_norm_4d_shape[kIndex2], l2_norm_4d_shape[kIndex3]),
      kernel_name_ + " running cudnnSetTensor4dDescriptor failed.")
  } else {
    CudnnSetTensorNdDescriptor(l2_norm_output_shape_, outputC_descriptor_, data_type_, kernel_name_);
  }
}

template <typename T, typename S>
void ClipByNormGpuKernelMod<T, S>::ChoseCudnnReduceTensorOp() {
  // Using `CUDNN_REDUCE_TENSOR_NORM2` to achieve `L2_norm` calculation and using float32 as input data type.
  reduce_tensor_op_ = CUDNN_REDUCE_TENSOR_NORM2;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor_, reduce_tensor_op_, CUDNN_DATA_FLOAT, nan_prop_,
                                   reduce_indices_, CUDNN_32BIT_INDICES),
    kernel_name_ + " running cudnnSetReduceTensorDescriptor failed.");
}

template <typename T, typename S>
void ClipByNormGpuKernelMod<T, S>::ResetResource() {
  // Reset resource for cudnnTensorReduce
  cudnn_handle_ = nullptr;
  data_type_ = CUDNN_DATA_FLOAT;
  nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
  reduce_tensor_op_ = CUDNN_REDUCE_TENSOR_NORM2;
  reduce_indices_ = CUDNN_REDUCE_TENSOR_NO_INDICES;
  reduce_tensor_descriptor_ = nullptr;
  inputA_descriptor_ = nullptr;
  outputC_descriptor_ = nullptr;
  // Reset member variables
  all_match_ = true;
  clip_norm_need_broadcast_ = false;
  epsilon_ = 0.000001f;
  x_dim_ = 0;
  x_size_ = 0;
  clip_norm_size_ = 0;
  l2_norm_output_size_ = 0;
  l2_norm_workspace_size_ = 0;
  output_size_ = 0;
  axis_.clear();
  x_shape_.clear();
  l2_norm_output_shape_.clear();
  clip_norm_shape_.clear();
  l2_norm_lhs_shape_.clear();
  l2_norm_rhs_shape_.clear();
  l2_norm_ouths_shape_.clear();
  clip_norm_rhs_shape_.clear();
  output_shape_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
}
}  // namespace kernel
}  // namespace mindspore
