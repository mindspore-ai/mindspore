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

#include "plugin/device/cpu/kernel/fmax_cpu_kernel.h"

#include <algorithm>
#include <utility>

#include "mindspore/core/ops/fmax.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/utils/cpu_utils.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kShapeIndexZero = 0;
constexpr auto kShapeIndex1st = 1;
constexpr auto kShapeIndex2nd = 2;
constexpr auto kShapeIndex3rd = 3;
constexpr auto kShapeIndex4th = 4;
constexpr auto kShapeIndex5th = 5;
constexpr auto kShapeIndex6th = 6;
constexpr size_t kFmaxInputsNum = 2;
constexpr size_t kFmaxOutputsNum = 1;
}  // namespace

bool FmaxCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Fmax>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast Fmax ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int FmaxCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }
  input_x_shape_ = inputs[0]->GetShapeVector();
  input_y_shape_ = inputs[1]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  output_num_ = 1;
  TypeId input_x_dtype = inputs[0]->GetDtype();
  TypeId input_y_dtype = inputs[1]->GetDtype();
  size_t max_input_shape_size =
    input_x_shape_.size() > input_y_shape_.size() ? input_x_shape_.size() : input_y_shape_.size();
  for (size_t i = 0; i < output_shape_.size(); i++) {
    output_num_ *= static_cast<size_t>(output_shape_[i]);
  }
  if ((input_x_shape_.size() == 0 && input_y_shape_.size() != 0) ||
      (input_x_shape_.size() != 0 && input_y_shape_.size() == 0)) {
    InitInputTensorAndScalar(max_input_shape_size);
  } else if (max_input_shape_size == output_shape_.size() && output_shape_.size() != 0) {
    InitInputTensors(input_x_dtype, input_y_dtype);
  }
  return 0;
}

void FmaxCpuKernelMod::InitInputTensorAndScalar(size_t max_input_shape_size) {
  if (max_input_shape_size != output_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of output tensor must be equal to the max "
                         "dimension of inputs, but got the dimension of output tensor: "
                      << output_shape_.size() << " and the max dimension of inputs: " << max_input_shape_size;
  }
  need_broadcast_ = false;
}

void FmaxCpuKernelMod::InitInputTensors(TypeId input_x_dtype, TypeId input_y_dtype) {
  if (input_x_dtype == kNumberTypeBool && input_y_dtype == kNumberTypeBool) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input tensor types can not be both bool.";
  }
  // Check if the shape needs to be broadcast
  need_broadcast_ = IsBroadcast();
  if (need_broadcast_) {
    InitTensorBroadcastShape();
  }
}

template <typename T>
bool FmaxCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) const {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kFmaxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kFmaxOutputsNum, kernel_name_);

  T *input_x_ = reinterpret_cast<T *>(inputs[0]->addr);
  T *input_y_ = reinterpret_cast<T *>(inputs[1]->addr);
  T *output_ = reinterpret_cast<T *>(outputs[0]->addr);
  BroadcastArith(input_x_, input_y_, output_);
  return true;
}

template <typename T>
void FmaxCpuKernelMod::BroadcastArith(const T *input_x, const T *input_y, T *output) const {
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_y);
  MS_EXCEPTION_IF_NULL(output);
  // bool need_broadcast = false;
  if (need_broadcast_) {
    BroadcastArithKernel(broadcast_input_x_shape_[kShapeIndexZero], broadcast_input_x_shape_[kShapeIndex1st],
                         broadcast_input_x_shape_[kShapeIndex2nd], broadcast_input_x_shape_[kShapeIndex3rd],
                         broadcast_input_x_shape_[kShapeIndex4th], broadcast_input_x_shape_[kShapeIndex5th],
                         broadcast_input_x_shape_[kShapeIndex6th], broadcast_input_y_shape_[kShapeIndexZero],
                         broadcast_input_y_shape_[kShapeIndex1st], broadcast_input_y_shape_[kShapeIndex2nd],
                         broadcast_input_y_shape_[kShapeIndex3rd], broadcast_input_y_shape_[kShapeIndex4th],
                         broadcast_input_y_shape_[kShapeIndex5th], broadcast_input_y_shape_[kShapeIndex6th],
                         broadcast_output_shape_[kShapeIndexZero], broadcast_output_shape_[kShapeIndex1st],
                         broadcast_output_shape_[kShapeIndex2nd], broadcast_output_shape_[kShapeIndex3rd],
                         broadcast_output_shape_[kShapeIndex4th], broadcast_output_shape_[kShapeIndex5th],
                         broadcast_output_shape_[kShapeIndex6th], input_x, input_y, output);
  } else {
    if (input_x_shape_.size() == 0 || input_y_shape_.size() == 0) {
      BroadcastArithOneScalarOneTensor(input_x, input_y, output);
    } else {
      BroadcastArithTensors(input_x, input_y, output);
    }
  }
}

bool FmaxCpuKernelMod::IsBroadcast() const {
  if (input_x_shape_.size() != input_y_shape_.size()) {
    return true;
  }
  for (size_t i = 0; i < input_x_shape_.size(); i++) {
    if (input_x_shape_[i] != input_y_shape_[i]) {
      return true;
    }
  }
  return false;
}

void FmaxCpuKernelMod::InitTensorBroadcastShape() {
  if (output_shape_.size() > max_dims_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of output must be less than or "
                         "equal to 7, but got "
                      << output_shape_.size() << ".";
  }
  broadcast_input_x_shape_.resize(max_dims_, 1);
  broadcast_input_y_shape_.resize(max_dims_, 1);
  broadcast_output_shape_.resize(max_dims_, 1);
  for (size_t i = 0; i < output_shape_.size(); i++) {
    broadcast_output_shape_[i] = static_cast<size_t>(output_shape_[i]);
  }
  int input_x_dim_offset = output_shape_.size() - input_x_shape_.size();
  for (size_t j = 0; j < input_x_shape_.size(); j++) {
    broadcast_input_x_shape_[j + IntToSize(input_x_dim_offset)] = static_cast<size_t>(input_x_shape_[j]);
    input_x_num_ *= static_cast<size_t>(input_x_shape_[j]);
  }
  int input_y_dim_offset = output_shape_.size() - input_y_shape_.size();
  for (size_t k = 0; k < input_y_shape_.size(); k++) {
    if (need_broadcast_) {
      broadcast_input_y_shape_[k + IntToSize(input_y_dim_offset)] = static_cast<size_t>(input_y_shape_[k]);
      input_y_num_ *= static_cast<size_t>(input_y_shape_[k]);
    }
  }
}

// FmaxFunc
template <typename T>
T FmaxCpuKernelMod::FmaxFunc(const T &lhs, const T &rhs) const {
  if (IsNan(lhs)) {
    return rhs;
  } else if (IsNan(rhs)) {
    return lhs;
  } else {
    return lhs > rhs ? lhs : rhs;
  }
}

// Broadcast comparison
int64_t FmaxCpuKernelMod::Index(const int64_t &index, const int64_t &dim) const { return dim == 1 ? 0 : index; }

// Broadcast Arithmetic
template <typename T>
void FmaxCpuKernelMod::BroadcastArithKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                            const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                            const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                            const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                            const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                            const size_t d6, const T *input_x, const T *input_y, T *output) const {
  for (size_t pos = 0; pos < output_num_; pos++) {
    auto pos_signed = SizeToLong(pos);
    size_t i = pos_signed / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos_signed / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos_signed / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos_signed / (d4 * d5 * d6) % d3;
    size_t m = pos_signed / (d5 * d6) % d4;
    size_t n = pos_signed / d6 % d5;
    size_t o = pos_signed % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    output[pos] = FmaxFunc(input_x[LongToSize(l_index)], input_y[LongToSize(r_index)]);
  }
}

template <typename T>
void FmaxCpuKernelMod::BroadcastArithOneScalarOneTensor(const T *input_x, const T *input_y, T *output) const {
  if (input_x_shape_.size() == 0) {
    for (size_t i = 0; i < output_num_; ++i) {
      output[i] = FmaxFunc(input_x[0], input_y[i]);
    }
  } else {
    for (size_t i = 0; i < output_num_; ++i) {
      output[i] = FmaxFunc(input_x[i], input_y[0]);
    }
  }
}

template <typename T>
void FmaxCpuKernelMod::BroadcastArithTensors(const T *input_x, const T *input_y, T *output) const {
  for (size_t i = 0; i < output_num_; ++i) {
    output[i] = FmaxFunc(input_x[i], input_y[i]);
  }
}

const std::vector<std::pair<KernelAttr, FmaxCpuKernelMod::KernelRunFunc>> &FmaxCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, FmaxCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &FmaxCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &FmaxCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &FmaxCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &FmaxCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &FmaxCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

std::vector<KernelAttr> FmaxCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  };
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Fmax, FmaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
