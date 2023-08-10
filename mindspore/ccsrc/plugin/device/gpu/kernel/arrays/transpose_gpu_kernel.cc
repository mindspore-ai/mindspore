/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/transpose_gpu_kernel.h"
#include "ops/transpose.h"
#include "kernel/kernel_get_value.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

constexpr size_t kPermInputNum = 2;
constexpr size_t kDimSize4 = 4;
constexpr size_t kAxisZero = 0;
constexpr size_t kAxis1st = 1;
constexpr size_t kAxis2nd = 2;
constexpr size_t kAxis3rd = 3;
constexpr size_t kAxisIndexZero = 0;
constexpr size_t kAxisIndex1st = 1;
constexpr size_t kAxisIndex2nd = 2;
constexpr size_t kAxisIndex3rd = 3;

#define OP_REGISTER(INPUTX, PERM, OUTPUT, T)                                    \
  {                                                                             \
    KernelAttr().AddInputAttr(INPUTX).AddInputAttr(PERM).AddOutputAttr(OUTPUT), \
      &TransposeGpuKernelMod::LaunchKernel<T>                                   \
  }

const std::vector<std::pair<KernelAttr, TransposeGpuKernelMod::KernelRunFunc>> &TransposeGpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, TransposeGpuKernelMod::KernelRunFunc>> func_list = {
    OP_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, kNumberTypeComplex64, Complex<float>),
    OP_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, kNumberTypeComplex128, Complex<double>),
    OP_REGISTER(kNumberTypeBool, kNumberTypeInt32, kNumberTypeBool, bool),
    OP_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, double),
    OP_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, float),
    OP_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, half),
    OP_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, int64_t),
    OP_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int32_t),
    OP_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt16, int16_t),
    OP_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, int8_t),
    OP_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, uint8_t),
    OP_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeUInt16, uint16_t),
    OP_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeUInt32, uint32_t),
    OP_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeUInt64, uint64_t),
    OP_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, kNumberTypeComplex64, Complex<float>),
    OP_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, kNumberTypeComplex128, Complex<double>),
    OP_REGISTER(kNumberTypeBool, kNumberTypeInt64, kNumberTypeBool, bool),
    OP_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, double),
    OP_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, float),
    OP_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, half),
    OP_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t),
    OP_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, int32_t),
    OP_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt16, int16_t),
    OP_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, int8_t),
    OP_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, uint8_t),
    OP_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeUInt16, uint16_t),
    OP_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeUInt32, uint32_t),
    OP_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeUInt64, uint64_t),
  };
  return func_list;
}

template <typename T>
bool TransposeGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);

  size_t size = SizeOf(input_shape_);
  if (is_copy_) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(output, input, size * sizeof(T), cudaMemcpyDeviceToDevice,
                                                       reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                       "For '" << kernel_name_ << "', cudaMemcpyAsync input to output failed.");
    return true;
  }
  auto status = CalTranspose<T, false>(size, input, info_, output, reinterpret_cast<cudaStream_t>(stream_ptr_));

  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

void TransposeGpuKernelMod::GetPermValue(const std::vector<int64_t> &perm, std::vector<int32_t> *input_perm) {
  input_perm->clear();
  for (size_t j = 0; j < perm.size(); j++) {
    auto p = (perm[j] >= 0) ? perm[j] : (perm.size() + perm[j]);
    if (p < 0) {
      MS_LOG(EXCEPTION) << "the perm value must be in [-" << perm.size() << ", " << (perm.size() - 1) << "], but got "
                        << perm;
    }
    input_perm->push_back(p);
  }
}

bool TransposeGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  size_t input_num = inputs.size();
  size_t output_num = outputs.size();
  kernel_name_ = base_operator->name();
  if (input_num != kPermInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be " << kPermInputNum << ", but got "
                      << input_num;
  }
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
  }
  return true;
}

bool TransposeGpuKernelMod::IsCopy(const std::vector<int32_t> &perm) {
  int32_t index = 0;
  return !(std::any_of(perm.begin(), perm.end(), [&](int32_t x) { return x != index++; }));
}

int TransposeGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  std::vector<int64_t> perm;
  std::vector<int32_t> input_perm;
  if (TryGetIntValue(inputs, kAxisIndex1st, kernel_name_, &perm)) {
    GetPermValue(perm, &input_perm);
  }
  auto input_shape = inputs[kAxisIndexZero]->GetDeviceShapeAdaptively();
  shape_size_ = input_shape.size();
  if (shape_size_ > transpose_max_dimension) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output cannot be greater than "
                      << transpose_max_dimension << ", but got " << shape_size_;
  }
  SimplifyTranspose(input_shape, input_perm, &input_shape_, &input_perm_);
  info_.input_shape = input_shape_;
  info_.perm = input_perm_;
  is_copy_ = IsCopy(input_perm_);
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Transpose, TransposeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
