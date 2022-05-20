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

#include "plugin/device/gpu/kernel/arrays/tensor_scatter_arithmetic_gpu_kernel.h"
#include "mindspore/core/ops/base_operator.h"
#include "mindspore/core/abstract/utils.h"

namespace mindspore {
namespace kernel {
#define TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(T_DT, S_DT, T, S)                             \
  KernelAttr().AddInputAttr(T_DT).AddInputAttr(S_DT).AddInputAttr(T_DT).AddOutputAttr(T_DT), \
    &TensorScatterArithmeticGpuKernelMod::LaunchKernel<T, S>

void TensorScatterArithmeticGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(input_size_);
  input_size_list_.push_back(indices_size_);
  input_size_list_.push_back(update_size_);
  output_size_list_.push_back(output_size_);
}

void TensorScatterArithmeticGpuKernelMod::FreeResource() {
  if (indices_stride_ != nullptr) {
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(indices_stride_);
    indices_stride_ = nullptr;
  }

  if (work_shape_ != nullptr) {
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(work_shape_);
    work_shape_ = nullptr;
  }
}

void TensorScatterArithmeticGpuKernelMod::ResetResource() {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  vec_work_shape_.clear();
}

bool TensorScatterArithmeticGpuKernelMod::GetOpTypeAndFuncType(const BaseOperatorPtr &base_operator) {
  static const std::map<std::string, TensorScatterArithmeticFunctionType> kTensorScatterOpTypeMap = {
    {"TensorScatterUpdate", TENSOR_SCATTER_FUNC_UPDATE}, {"TensorScatterMin", TENSOR_SCATTER_FUNC_MIN},
    {"TensorScatterMax", TENSOR_SCATTER_FUNC_MAX},       {"TensorScatterAdd", TENSOR_SCATTER_FUNC_ADD},
    {"TensorScatterSub", TENSOR_SCATTER_FUNC_SUB},       {"TensorScatterMul", TENSOR_SCATTER_FUNC_MUL},
    {"TensorScatterDiv", TENSOR_SCATTER_FUNC_DIV}};
  auto op_type_iter = kTensorScatterOpTypeMap.find(kernel_name_);
  if (op_type_iter == kTensorScatterOpTypeMap.end()) {
    MS_LOG(ERROR) << "Only support these tensor_scatter function: TensorScatterUpdate, TensorScatterMin, "
                     "TensorScatterMax, TensorScatterAdd, TensorScatterSub, TensorScatterMul or TensorScatterDiv "
                     "currently, but got "
                  << kernel_name_;
    return false;
  }
  op_func_type_ = op_type_iter->second;

  /* Get kernel_ptr_ by kernel name */
  auto prim = base_operator->GetPrim();
  static const std::map<std::string, BaseOperatorPtr> kTensorScatterOpPrimitiveMap = {
    {"TensorScatterUpdate", std::make_shared<ops::TensorScatterUpdate>(prim)},
    {"TensorScatterMin", std::make_shared<ops::TensorScatterMin>(prim)},
    {"TensorScatterMax", std::make_shared<ops::TensorScatterMax>(prim)},
    {"TensorScatterAdd", std::make_shared<ops::TensorScatterAdd>(prim)},
    {"TensorScatterSub", std::make_shared<ops::TensorScatterSub>(prim)},
    {"TensorScatterMul", std::make_shared<ops::TensorScatterMul>(prim)},
    {"TensorScatterDiv", std::make_shared<ops::TensorScatterDiv>(prim)}};
  auto op_prim_iter = kTensorScatterOpPrimitiveMap.find(kernel_name_);
  if (op_prim_iter == kTensorScatterOpPrimitiveMap.end()) {
    MS_LOG(ERROR) << "Only support these tensor_scatter function: TensorScatterUpdate, TensorScatterMin, "
                     "TensorScatterMax, TensorScatterAdd, TensorScatterSub, TensorScatterMul or TensorScatterDiv "
                     "currently, but got "
                  << kernel_name_;
    return false;
  }
  kernel_ptr_ = op_prim_iter->second;
  return true;
}

void TensorScatterArithmeticGpuKernelMod::UpdateSize() {
  input_size_ = data_unit_size_;
  for (const auto &shape_item : input_shapes_) {
    input_size_ *= shape_item;
  }
  indices_size_ = indices_unit_size_;
  for (const auto &shape_item : indices_shapes_) {
    indices_size_ *= shape_item;
  }
  update_size_ = data_unit_size_;
  for (const auto &shape_item : update_shapes_) {
    update_size_ *= shape_item;
  }
  output_size_ = data_unit_size_;
  for (const auto &shape_item : output_shapes_) {
    output_size_ *= shape_item;
  }

  // calculate indices dim 0/1
  indices_dim_0_ = indices_shapes_[0];
  indices_dim_1_ = indices_shapes_[indices_shapes_.size() - 1];

  // calculate block_size
  block_size_ = 1;
  for (size_t i = indices_dim_1_; i < output_shapes_.size(); i++) {
    block_size_ *= output_shapes_[i];
  }

  // calculate indices_stride
  vec_indices_stride_.resize(indices_dim_1_, 0);
  vec_indices_stride_[indices_dim_1_ - 1] = block_size_;

  for (size_t i = indices_dim_1_ - 1; i > 0; --i) {
    vec_indices_stride_[i - 1] = vec_indices_stride_[i] * output_shapes_[i];
  }
}

bool TensorScatterArithmeticGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Got empty inputs or outputs, which is invalid.";
    return false;
  }

  kernel_name_ = base_operator->name();
  auto ret = GetOpTypeAndFuncType(base_operator);
  if (!ret) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' get op type and function type failed.";
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  indices_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).first);
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);

  return true;
}

int TensorScatterArithmeticGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  FreeResource();
  ResetResource();

  memcpy_flag_ = false;

  input_shapes_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                      inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  indices_shapes_ = std::vector<size_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                        inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  update_shapes_ = std::vector<size_t>(inputs.at(kIndex2)->GetDeviceShapeAdaptively().begin(),
                                       inputs.at(kIndex2)->GetDeviceShapeAdaptively().end());
  output_shapes_ = std::vector<size_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                       outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());

  std::vector<size_t> shape_me = input_shapes_;
  (void)std::transform(shape_me.begin(), shape_me.end(), std::back_inserter(vec_work_shape_),
                       [](const size_t &value) { return static_cast<size_t>(value); });

  UpdateSize();

  const size_t indices_len = indices_unit_size_ * vec_indices_stride_.size();
  indices_stride_ = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(indices_len);
  if (indices_stride_ == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the memory alloc of indices_stride_ must be successful, but failed, got size: "
                      << indices_len;
  }

  const size_t vec_work_len = indices_unit_size_ * vec_work_shape_.size();
  work_shape_ = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(vec_work_len);
  if (work_shape_ == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the memory alloc of work_shape_ must be successful, but failed, got size: "
                      << vec_work_len;
  }

  InitSizeLists();

  return KRET_OK;
}

template <typename T, typename S>
bool TensorScatterArithmeticGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                       const std::vector<AddressPtr> &workspace,
                                                       const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  VARIABLE_NOT_USED(workspace);
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices = GetDeviceAddress<S>(inputs, kIndex1);
  T *update = GetDeviceAddress<T>(inputs, kIndex2);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);

  if (!memcpy_flag_) {
    const size_t indices_len = indices_unit_size_ * vec_indices_stride_.size();
    std::vector<S> vec_indices_stride_s = std::vector<S>(vec_indices_stride_.begin(), vec_indices_stride_.end());
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(indices_stride_, vec_indices_stride_s.data(), indices_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpy failed in TensorScatterArithmeticGpuKernelMod::Launch.");

    const size_t vec_work_len = indices_unit_size_ * vec_work_shape_.size();
    std::vector<S> vec_work_shape_s = std::vector<S>(vec_work_shape_.begin(), vec_work_shape_.end());
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(work_shape_, vec_work_shape_s.data(), vec_work_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpy failed in TensorScatterArithmeticGpuKernelMod::Launch.");
    memcpy_flag_ = true;
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output, input, input_size_, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpy output failed");

  TensorScatterArithmetic(op_func_type_, input, indices, update, output, block_size_, update_size_ / data_unit_size_,
                          output_size_ / data_unit_size_, indices_dim_0_, indices_dim_1_,
                          reinterpret_cast<S *>(indices_stride_), reinterpret_cast<S *>(work_shape_),
                          reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, TensorScatterArithmeticGpuKernelMod::TensorScatterArithmeticFunc>>
  TensorScatterArithmeticGpuKernelMod::func_list_ = {
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, half, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeInt8, kNumberTypeInt32, char, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, uchar, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeInt32, kNumberTypeInt32, int, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeBool, kNumberTypeInt32, bool, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, half, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeInt8, kNumberTypeInt64, char, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, uchar, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeInt32, kNumberTypeInt64, int, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeBool, kNumberTypeInt64, bool, int64_t)},
};

std::vector<KernelAttr> TensorScatterArithmeticGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, TensorScatterArithmeticFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TensorScatterUpdate, TensorScatterArithmeticGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TensorScatterMin, TensorScatterArithmeticGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TensorScatterMax, TensorScatterArithmeticGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TensorScatterAdd, TensorScatterArithmeticGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TensorScatterSub, TensorScatterArithmeticGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TensorScatterMul, TensorScatterArithmeticGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TensorScatterDiv, TensorScatterArithmeticGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
