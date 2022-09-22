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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

constexpr size_t kDynamicPermInputNum = 2;
constexpr size_t kDimSize4 = 4;
constexpr size_t kAxisZero = 0;
constexpr size_t kAxis1st = 1;
constexpr size_t kAxis2nd = 2;
constexpr size_t kAxis3rd = 3;
constexpr size_t kAxisIndexZero = 0;
constexpr size_t kAxisIndex1st = 1;
constexpr size_t kAxisIndex2nd = 2;
constexpr size_t kAxisIndex3rd = 3;

#define STATIC_REGISTER(INPUTX, OUTPUT, T) \
  { KernelAttr().AddInputAttr(INPUTX).AddOutputAttr(OUTPUT), &TransposeGpuKernelMod::LaunchKernel<T> }

#define DYN_REGISTER(INPUTX, PERM, OUTPUT, T)                                   \
  {                                                                             \
    KernelAttr().AddInputAttr(INPUTX).AddInputAttr(PERM).AddOutputAttr(OUTPUT), \
      &TransposeGpuKernelMod::LaunchKernel<T>                                   \
  }

const std::vector<std::pair<KernelAttr, TransposeGpuKernelMod::KernelRunFunc>> &TransposeGpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, TransposeGpuKernelMod::KernelRunFunc>> func_list = {
    STATIC_REGISTER(kNumberTypeComplex64, kNumberTypeComplex64, Complex<float>),
    STATIC_REGISTER(kNumberTypeComplex128, kNumberTypeComplex128, Complex<double>),
    STATIC_REGISTER(kNumberTypeBool, kNumberTypeBool, bool),
    STATIC_REGISTER(kNumberTypeFloat64, kNumberTypeFloat64, double),
    STATIC_REGISTER(kNumberTypeFloat32, kNumberTypeFloat32, float),
    STATIC_REGISTER(kNumberTypeFloat16, kNumberTypeFloat16, half),
    STATIC_REGISTER(kNumberTypeInt64, kNumberTypeInt64, int64_t),
    STATIC_REGISTER(kNumberTypeInt32, kNumberTypeInt32, int32_t),
    STATIC_REGISTER(kNumberTypeInt16, kNumberTypeInt16, int16_t),
    STATIC_REGISTER(kNumberTypeInt8, kNumberTypeInt8, int8_t),
    STATIC_REGISTER(kNumberTypeUInt8, kNumberTypeUInt8, uint8_t),
    STATIC_REGISTER(kNumberTypeUInt16, kNumberTypeUInt16, uint16_t),
    STATIC_REGISTER(kNumberTypeUInt32, kNumberTypeUInt32, uint32_t),
    STATIC_REGISTER(kNumberTypeUInt64, kNumberTypeUInt64, uint64_t),
    DYN_REGISTER(kNumberTypeComplex64, kNumberTypeInt32, kNumberTypeComplex64, Complex<float>),
    DYN_REGISTER(kNumberTypeComplex128, kNumberTypeInt32, kNumberTypeComplex128, Complex<double>),
    DYN_REGISTER(kNumberTypeBool, kNumberTypeInt32, kNumberTypeBool, bool),
    DYN_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeFloat64, double),
    DYN_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, float),
    DYN_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, half),
    DYN_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, int64_t),
    DYN_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int32_t),
    DYN_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt16, int16_t),
    DYN_REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt8, int8_t),
    DYN_REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeUInt8, uint8_t),
    DYN_REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeUInt16, uint16_t),
    DYN_REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeUInt32, uint32_t),
    DYN_REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeUInt64, uint64_t),
    DYN_REGISTER(kNumberTypeComplex64, kNumberTypeInt64, kNumberTypeComplex64, Complex<float>),
    DYN_REGISTER(kNumberTypeComplex128, kNumberTypeInt64, kNumberTypeComplex128, Complex<double>),
    DYN_REGISTER(kNumberTypeBool, kNumberTypeInt64, kNumberTypeBool, bool),
    DYN_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeFloat64, double),
    DYN_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, float),
    DYN_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, half),
    DYN_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t),
    DYN_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, int32_t),
    DYN_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt16, int16_t),
    DYN_REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt8, int8_t),
    DYN_REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeUInt8, uint8_t),
    DYN_REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeUInt16, uint16_t),
    DYN_REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeUInt32, uint32_t),
    DYN_REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeUInt64, uint64_t),
  };
  return func_list;
}

template <typename T>
bool TransposeGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);
  size_t *input_shape = GetDeviceAddress<size_t>(workspace, 0);
  size_t *input_axis = GetDeviceAddress<size_t>(workspace, 1);

  if (is_dynamic_perm_ && !get_dynamic_perm_value_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', fail to get value of the dynamic perm!";
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(input_shape, &input_shape_[0], workspace_size_, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpyAsync input_shape failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(input_axis, &input_perm_[0], workspace_size_, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpyAsync input_axis failed");
  size_t size = SizeOf(input_shape_);
  size_t *h_input_shape = reinterpret_cast<size_t *>(&input_shape_[0]);
  size_t *h_input_axis = &input_perm_[0];
  if (shape_size_ == kDimSize4 && h_input_axis[kAxisIndexZero] == kAxisZero &&
      h_input_axis[kAxisIndex1st] == kAxis3rd && h_input_axis[kAxisIndex2nd] == kAxis1st &&
      h_input_axis[kAxisIndex3rd] == kAxis2nd) {
    // nhwc->nchw: 0,3,1,2
    CalNHWC2NCHWInterface(size, shape_size_, input, h_input_shape, h_input_axis, input_shape, input_axis, output,
                          reinterpret_cast<cudaStream_t>(stream_ptr_));
  } else if (shape_size_ == kDimSize4 && h_input_axis[kAxisIndexZero] == kAxisZero &&
             h_input_axis[kAxisIndex1st] == kAxis2nd && h_input_axis[kAxisIndex2nd] == kAxis3rd &&
             h_input_axis[kAxisIndex3rd] == kAxis1st) {
    // nchw->nhwc: 0,2,3,1
    CalNCHW2NHWCInterface(size, shape_size_, input, h_input_shape, h_input_axis, input_shape, input_axis, output,
                          reinterpret_cast<cudaStream_t>(stream_ptr_));
  } else {
    CalTranspose(size, input, input_shape, input_axis, shape_size_, output,
                 reinterpret_cast<cudaStream_t>(stream_ptr_));
  }
  return true;
}

void TransposeGpuKernelMod::GetPermValue(const std::vector<int64_t> &perm) {
  for (size_t j = 0; j < perm.size(); j++) {
    auto p = (perm[j] >= 0) ? perm[j] : (perm.size() + perm[j]);
    if (p < 0) {
      MS_LOG(EXCEPTION) << "the perm value must be in [-" << perm.size() << ", " << (perm.size() - 1) << "], but got "
                        << perm;
    }
    input_perm_.push_back(p);
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
  if (input_num != 1 && input_num != kDynamicPermInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1 or " << kDynamicPermInputNum
                      << ", but got " << input_num;
  }
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
  }
  if (input_num == kDynamicPermInputNum) {
    is_dynamic_perm_ = true;
    return true;
  }

  auto attr = base_operator->GetPrim()->GetAttr(kAttrPerm);
  if (attr == nullptr) {
    MS_LOG(ERROR) << "The attr \"perm\" is not found in kernel 'Transpose'.";
    return false;
  }
  std::vector<int64_t> perm;
  if (attr->isa<tensor::Tensor>()) {
    perm = CheckAndConvertUtils::CheckTensorIntValue("perm", attr, kernel_name_);
  } else {
    perm = CheckAndConvertUtils::CheckIntOrTupleInt("perm", attr, kernel_name_);
  }
  GetPermValue(perm);
  return true;
}

int TransposeGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  std::vector<int64_t> perm;
  if (GetDynamicAttrIntValue(inputs, kAxisIndex1st, inputsOnHost, kernel_name_, &perm)) {
    GetPermValue(perm);
    get_dynamic_perm_value_ = true;
  }
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs[kAxisIndexZero]->GetDeviceShapeAdaptively();
  shape_size_ = input_shape_.size();
  if (shape_size_ > TRANSPOSE_MAX_DIMENSION) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output cannot be greater than "
                      << TRANSPOSE_MAX_DIMENSION << ", but got " << shape_size_;
  }

  workspace_size_ = shape_size_ * sizeof(size_t);
  workspace_size_list_.push_back(workspace_size_);
  workspace_size_list_.push_back(workspace_size_);
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Transpose, TransposeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
