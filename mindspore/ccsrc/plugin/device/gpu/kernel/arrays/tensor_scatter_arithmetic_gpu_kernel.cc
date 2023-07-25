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
#include "mindspore/core/abstract/utils.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/base_operator.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
#define TENSOR_SCATTER_ARITHMETIC_GPU_REGISTER(IN_DT0, IN_DT1, IN_DT2, OUT_DT0, T, S)                 \
  KernelAttr().AddInputAttr(IN_DT0).AddInputAttr(IN_DT1).AddInputAttr(IN_DT2).AddOutputAttr(OUT_DT0), \
    &TensorScatterArithmeticGpuKernelMod::LaunchKernel<T, S>

constexpr auto kTensorScatterUpdate = "TensorScatterUpdate";

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
  indices_dim_0_ = static_cast<size_t>(indices_shape_[0]);
  indices_dim_1_ = static_cast<size_t>(indices_shape_[indices_shape_.size() - 1]);
  // Calculate block_size
  int64_t block_size = 1;
  for (size_t i = indices_dim_1_; i < output_shape_.size(); i++) {
    block_size *= output_shape_[i];
  }
  block_size_ = static_cast<size_t>(block_size);
  // Calculate indices_stride
  vec_indices_stride_.resize(indices_dim_1_, 0);
  vec_indices_stride_[indices_dim_1_ - 1] = block_size;
  for (size_t i = indices_dim_1_ - 1; i > 0; --i) {
    vec_indices_stride_[i - 1] = vec_indices_stride_[i] * output_shape_[i];
  }
}

bool TensorScatterArithmeticGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
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
  data_unit_size_ = abstract::TypeIdSize(inputs.at(kIndex0)->GetDtype());
  indices_unit_size_ = abstract::TypeIdSize(inputs.at(kIndex1)->GetDtype());
  return true;
}

int TensorScatterArithmeticGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  indices_shape_ = inputs.at(kIndex1)->GetShapeVector();
  update_shape_ = inputs.at(kIndex2)->GetShapeVector();

  input_size_ = static_cast<size_t>(
    std::accumulate(input_shape_.begin(), input_shape_.end(), int64_t(1), std::multiplies<int64_t>()));
  update_size_ = static_cast<size_t>(
    std::accumulate(update_shape_.begin(), update_shape_.end(), int64_t(1), std::multiplies<int64_t>()));

  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
  output_size_ =
    static_cast<size_t>(std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<int64_t>()));
  UpdateSize();
  return KRET_OK;
}
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S>
bool TensorScatterArithmeticGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                       const std::vector<AddressPtr> &workspace,
                                                       const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices = GetDeviceAddress<S>(inputs, kIndex1);
  T *update = GetDeviceAddress<T>(inputs, kIndex2);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  // vec_indices_stride and work_shape
  TensorScatterInfo<S> info;
  for (size_t i = 0; i < indices_dim_1_; ++i) {
    info.indices_stride[i] = static_cast<S>(vec_indices_stride_[i]);
    info.work_shape[i] = static_cast<S>(input_shape_[i]);
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output, input, input_size_ * data_unit_size_, cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpy output failed");
  if constexpr ((std::is_same_v<T, Complex<float>>) || (std::is_same_v<T, Complex<double>>)) {
    if (kernel_name_ == kTensorScatterUpdate) {
      auto status =
        CallTensorScatterUpdate(input, indices, update, output, block_size_, update_size_, output_size_, indices_dim_0_,
                                indices_dim_1_, info, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr_));
      CHECK_CUDA_STATUS(status, kernel_name_);
      return true;
    }
  } else {
    auto status = TensorScatterArithmetic(op_func_type_, input, indices, update, output, block_size_, update_size_,
                                          output_size_, indices_dim_0_, indices_dim_1_, info, device_id_,
                                          reinterpret_cast<cudaStream_t>(stream_ptr_));
    CHECK_CUDA_STATUS(status, kernel_name_);
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
