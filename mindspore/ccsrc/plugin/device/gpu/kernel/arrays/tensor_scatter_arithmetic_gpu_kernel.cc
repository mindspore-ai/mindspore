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
#include <functional>
#include "mindspore/core/ops/base_operator.h"
#include "mindspore/core/abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
#define TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(IN_DT0, IN_DT1, IN_DT2, OUT_DT0, T, S)                 \
  KernelAttr().AddInputAttr(IN_DT0).AddInputAttr(IN_DT1).AddInputAttr(IN_DT2).AddOutputAttr(OUT_DT0), \
    &TensorScatterArithmeticGpuKernelMod::LaunchKernel<T, S>

constexpr auto kTensorScatterUpdate = "TensorScatterUpdate";

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

bool TensorScatterArithmeticGpuKernelMod::GetOpType(const BaseOperatorPtr &base_operator) {
  static const std::map<std::string, TensorScatterArithmeticFunctionType> tensor_scatter_op_map = {
    {prim::kPrimTensorScatterUpdate->name(), TENSOR_SCATTER_FUNC_UPDATE},
    {prim::kPrimTensorScatterMin->name(), TENSOR_SCATTER_FUNC_MIN},
    {prim::kPrimTensorScatterMax->name(), TENSOR_SCATTER_FUNC_MAX},
    {prim::kPrimTensorScatterAdd->name(), TENSOR_SCATTER_FUNC_ADD},
    {prim::kPrimTensorScatterSub->name(), TENSOR_SCATTER_FUNC_SUB},
    {prim::kPrimTensorScatterMul->name(), TENSOR_SCATTER_FUNC_MUL},
    {prim::kPrimTensorScatterDiv->name(), TENSOR_SCATTER_FUNC_DIV}};
  auto op_type_iter = tensor_scatter_op_map.find(kernel_name_);
  if (op_type_iter == tensor_scatter_op_map.end()) {
    MS_LOG(ERROR) << "Only support these tensor_scatter function: TensorScatterUpdate, TensorScatterMin, "
                     "TensorScatterMax, TensorScatterAdd, TensorScatterSub, TensorScatterMul or TensorScatterDiv "
                     "currently, but got "
                  << kernel_name_;
    return false;
  }
  op_func_type_ = op_type_iter->second;
  return true;
}

void TensorScatterArithmeticGpuKernelMod::UpdateSize() {
  // Calculate indices dim 0/1
  indices_dim_0_ = indices_shape_[0];
  indices_dim_1_ = indices_shape_[indices_shape_.size() - 1];
  // Calculate block_size
  block_size_ = 1;
  for (size_t i = indices_dim_1_; i < output_shape_.size(); i++) {
    block_size_ *= output_shape_[i];
  }
  // Calculate indices_stride
  vec_indices_stride_.resize(indices_dim_1_, 0);
  vec_indices_stride_[indices_dim_1_ - 1] = block_size_;
  for (size_t i = indices_dim_1_ - 1; i > 0; --i) {
    vec_indices_stride_[i - 1] = vec_indices_stride_[i] * output_shape_[i];
  }
}

bool TensorScatterArithmeticGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!GetOpType(base_operator)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' it got op type and function type failed.";
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto type_id = kernel_attr.GetInputAttr(kIndex0).dtype;
  if ((type_id == kNumberTypeComplex64 || type_id == kNumberTypeComplex128) && (kernel_name_ != kTensorScatterUpdate)) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the data type of input args not supports Complex.";
    return false;
  }
  return true;
}

int TensorScatterArithmeticGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  FreeResource();
  workspace_size_list_.clear();
  vec_work_shape_.clear();
  memcpy_flag_ = false;
  data_unit_size_ = abstract::TypeIdSize(inputs.at(kIndex0)->GetDtype());
  indices_unit_size_ = abstract::TypeIdSize(inputs.at(kIndex1)->GetDtype());

  input_shape_.clear();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());

  indices_shape_.clear();
  auto indices_shape = inputs.at(kIndex1)->GetShapeVector();
  constexpr size_t k_min_indices_rank = 2;
  // To correspondence with other backend, We need to except ValueError, so just give exception info for dimension
  // checking.
  if (indices_shape.size() < k_min_indices_rank) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "' , the dimension of 'indices' must be at least 2, but got "
                             << indices_shape.size();
  }
  (void)std::transform(indices_shape.begin(), indices_shape.end(), std::back_inserter(indices_shape_), LongToSize);

  update_shape_.clear();
  auto update_shapes = inputs.at(kIndex2)->GetShapeVector();
  (void)std::transform(update_shapes.begin(), update_shapes.end(), std::back_inserter(update_shape_), LongToSize);
  update_size_ = std::accumulate(update_shape_.begin(), update_shape_.end(), size_t(1), std::multiplies<size_t>());

  output_shape_.clear();
  auto output_shapes = outputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(output_shapes.begin(), output_shapes.end(), std::back_inserter(output_shape_), LongToSize);
  output_size_ = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<size_t>());
  vec_work_shape_ = input_shape_;
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

  const auto indices_rank = indices_shape.size();
  const auto last_indices_value = LongToSize(indices_shape.back());
  const auto update_rank = update_shape_.size();
  constexpr size_t min_indices_rank = 2;
  slice_size_ = last_indices_value;
  batch_size_ = 1;
  inner_size_ = 1;

  for (size_t i = 0; i < update_rank; ++i) {
    if (i <= indices_rank - min_indices_rank) {
      batch_size_ *= LongToSize(indices_shape[i]);
    } else {
      inner_size_ *= LongToSize(update_shape_[i]);
    }
  }
  batch_strides_.resize(slice_size_);
  for (auto i = SizeToLong(slice_size_) - 1; i >= 0; --i) {
    auto dim = LongToSize(i);
    total_batch_size_ *= input_shape_[dim];
    if (dim == slice_size_ - 1) {
      batch_strides_[dim] = 1;
    } else {
      batch_strides_[dim] = batch_strides_[dim + 1] * input_shape_[dim + 1];
    }
  }

  return KRET_OK;
}
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename S>
void TensorScatterArithmeticGpuKernelMod::CheckIndicesValid(S *indices) {
  size_t total_indices_num =
    std::accumulate(indices_shape_.begin(), indices_shape_.end(), 1, std::multiplies<size_t>());
  size_t total_indices_bytes = total_indices_num * indices_unit_size_;
  S *host_indices = reinterpret_cast<S *>(malloc(total_indices_bytes));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(host_indices, indices, total_indices_bytes, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "TensorScatterArithmeticGpuKernelMod cudaMemcpy failed in TensorScatterArithmeticGpuKernelMod::CheckIndexValid.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "cudaStreamSynchronized failed");
  int64_t invalid_index_pos = -1;
  for (size_t i = 0; i < batch_size_; ++i) {
    size_t out_index = 0;
    for (size_t j = 0; j < slice_size_; ++j) {
      S idx_index = host_indices[SizeToLong(i) * slice_size_ + SizeToLong(j)];
      out_index += batch_strides_[j] * static_cast<size_t>(idx_index);
      if (idx_index < 0 || idx_index >= static_cast<S>(input_shape_[j])) {
        invalid_index_pos = SizeToLong(i * slice_size_);
        break;
      }
    }
    if (invalid_index_pos != -1) {
      break;
    }
  }
  if (invalid_index_pos != -1) {
    std::stringstream indices_ss;
    std::stringstream input_shape_ss;
    for (size_t i = 0; i < slice_size_; i++) {
      if (i > 0) {
        indices_ss << ", ";
        input_shape_ss << ", ";
      }
      indices_ss << std::to_string(host_indices[LongToSize(invalid_index_pos) + i]);
      input_shape_ss << std::to_string(input_shape_[i]);
    }
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the " << invalid_index_pos << "-th value of 'indices'["
                      << indices_ss.str() << "] is out of range[" << input_shape_ss.str() << "].";
  }
}

template <typename T, typename S>
bool TensorScatterArithmeticGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                       const std::vector<AddressPtr> &workspace,
                                                       const std::vector<AddressPtr> &outputs) {
  VARIABLE_NOT_USED(workspace);
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices = GetDeviceAddress<S>(inputs, kIndex1);
  T *update = GetDeviceAddress<T>(inputs, kIndex2);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);

  (void)CheckIndicesValid(indices);

  if (!memcpy_flag_) {
    const size_t indices_len = indices_unit_size_ * vec_indices_stride_.size();
    std::vector<S> vec_indices_stride_s = std::vector<S>(vec_indices_stride_.begin(), vec_indices_stride_.end());
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(indices_stride_, vec_indices_stride_s.data(), indices_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr_)),
      "TensorScatterArithmeticGpuKernelMod cudaMemcpy failed in TensorScatterArithmeticGpuKernelMod::Launch.");

    const size_t vec_work_len = indices_unit_size_ * vec_work_shape_.size();
    std::vector<S> vec_work_shape_s = std::vector<S>(vec_work_shape_.begin(), vec_work_shape_.end());
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(work_shape_, vec_work_shape_s.data(), vec_work_len, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr_)),
      "TensorScatterArithmeticGpuKernelMod cudaMemcpy failed in TensorScatterArithmeticGpuKernelMod::Launch.");
    memcpy_flag_ = true;
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output, input, input_size_ * data_unit_size_, cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpy output failed");

  if constexpr ((std::is_same_v<T, Complex<float>>) || (std::is_same_v<T, Complex<double>>)) {
    if (kernel_name_ == kTensorScatterUpdate) {
      CallTensorScatterUpdate(input, indices, update, output, block_size_, update_size_, output_size_, indices_dim_0_,
                              indices_dim_1_, reinterpret_cast<S *>(indices_stride_),
                              reinterpret_cast<S *>(work_shape_), device_id_,
                              reinterpret_cast<cudaStream_t>(stream_ptr_));
      return true;
    }
  } else {
    TensorScatterArithmetic(op_func_type_, input, indices, update, output, block_size_, update_size_, output_size_,
                            indices_dim_0_, indices_dim_1_, reinterpret_cast<S *>(indices_stride_),
                            reinterpret_cast<S *>(work_shape_), device_id_,
                            reinterpret_cast<cudaStream_t>(stream_ptr_));
  }
  return true;
}

const TensorScatterArithmeticGpuKernelMod::SupportList &TensorScatterArithmeticGpuKernelMod::GetFuncList() const {
  static const TensorScatterArithmeticGpuKernelMod::SupportList func_list = {
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, char,
                                            int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt16, kNumberTypeInt16,
                                            int16_t, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32,
                                            int32_t, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64,
                                            int64_t, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8,
                                            uint8_t, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeUInt16, kNumberTypeUInt16,
                                            uint16_t, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeUInt32, kNumberTypeUInt32,
                                            uint32_t, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeUInt64, kNumberTypeUInt64,
                                            uint64_t, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16,
                                            kNumberTypeFloat16, half, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32,
                                            kNumberTypeFloat32, float, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64,
                                            kNumberTypeFloat64, double, int)},

    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, char,
                                            int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt16, kNumberTypeInt16,
                                            int16_t, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32,
                                            int32_t, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64,
                                            int64_t, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8,
                                            uint8_t, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeUInt16, kNumberTypeUInt16,
                                            uint16_t, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeUInt32, kNumberTypeUInt32,
                                            uint32_t, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeUInt64, kNumberTypeUInt64,
                                            uint64_t, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16,
                                            kNumberTypeFloat16, half, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32,
                                            kNumberTypeFloat32, float, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64,
                                            kNumberTypeFloat64, double, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeBool, kNumberTypeInt32, kNumberTypeBool, kNumberTypeBool, bool,
                                            int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeBool, kNumberTypeInt64, kNumberTypeBool, kNumberTypeBool, bool,
                                            int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, kNumberTypeComplex64,
                                            kNumberTypeComplex64, Complex<float>, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, kNumberTypeComplex128,
                                            kNumberTypeComplex128, Complex<double>, int)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, kNumberTypeComplex64,
                                            kNumberTypeComplex64, Complex<float>, int64_t)},
    {TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, kNumberTypeComplex128,
                                            kNumberTypeComplex128, Complex<double>, int64_t)},
  };
  return func_list;
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
