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

#include "plugin/device/cpu/kernel/eigen/expand_cpu_kernel.h"
#include <algorithm>
#include "unsupported/Eigen/CXX11/Tensor"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kExpandInputsNum = 2;
const size_t kExpandOutputsNum = 1;
const size_t kNoBroadcastValue = 1;
const size_t kRank0 = 0;
const size_t kRank1 = 1;
const size_t kRank2 = 2;
const size_t kRank3 = 3;
const size_t kRank4 = 4;
const size_t kRank5 = 5;
const size_t kRank6 = 6;
const size_t kRank7 = 7;
const size_t kRank8 = 8;
}  // namespace

bool ExpandCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kExpandInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kExpandOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int ExpandCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  input_x_shape_ = LongVecToSizeVec(inputs[kIndex0]->GetDeviceShapeAdaptively());
  input_x_dtype_ = inputs[kIndex0]->GetDtype();
  input_shape_ = LongVecToSizeVec(outputs[kIndex0]->GetDeviceShapeAdaptively());
  output_y_shape_ = LongVecToSizeVec(outputs[kIndex0]->GetDeviceShapeAdaptively());
  return KRET_OK;
}

bool ExpandCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kExpandInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kExpandOutputsNum, kernel_name_);
  switch (input_x_dtype_) {
    case kNumberTypeFloat16:
      return ExpandCompute<float16>(inputs, outputs);
    case kNumberTypeFloat32:
      return ExpandCompute<float>(inputs, outputs);
    case kNumberTypeInt8:
      return ExpandCompute<int8_t>(inputs, outputs);
    case kNumberTypeInt32:
      return ExpandCompute<int32_t>(inputs, outputs);
    case kNumberTypeUInt8:
      return ExpandCompute<uint8_t>(inputs, outputs);
    default:
      MS_LOG(EXCEPTION) << "For " << kernel_name_
                        << ", the dtype of input `x` must in [float16, float32, int8, int32, uint8] "
                        << "but got " << TypeIdToType(input_x_dtype_)->ToString() << ".";
      return false;
  }
}

size_t ExpandCpuKernelMod::get_element_num(const std::vector<size_t> &shape) const {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  return size;
}

template <typename T>
bool ExpandCpuKernelMod::ExpandCompute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  size_t rank = static_cast<size_t>(output_y_shape_.size());
  switch (rank) {
    case kRank0: {
      T v0 = *(reinterpret_cast<const T *>(inputs[0]->addr));
      T *value_out = reinterpret_cast<T *>(outputs[0]->addr);
      *(value_out) = v0;
      return true;
    }
    case kRank1:
      return ExpandCalculate<kRank1, T>(inputs, outputs);
    case kRank2:
      return ExpandCalculate<kRank2, T>(inputs, outputs);
    case kRank3:
      return ExpandCalculate<kRank3, T>(inputs, outputs);
    case kRank4:
      return ExpandCalculate<kRank4, T>(inputs, outputs);
    case kRank5:
      return ExpandCalculate<kRank5, T>(inputs, outputs);
    case kRank6:
      return ExpandCalculate<kRank6, T>(inputs, outputs);
    case kRank7:
      return ExpandCalculate<kRank7, T>(inputs, outputs);
    case kRank8:
      return ExpandCalculate<kRank8, T>(inputs, outputs);
    default:
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the rank of output should not expand than 8 but got "
                        << std::to_string(rank) << ".";
      return false;
  }
}

template <size_t RANK, typename T>
bool ExpandCpuKernelMod::ExpandCalculate(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  size_t input_x_element_num = get_element_num(input_x_shape_);
  size_t output_y_element_num = get_element_num(output_y_shape_);

  (void)input_x_shape_.insert(input_x_shape_.begin(), RANK - input_x_shape_.size(), 1);
  input_x_bcast_.clear();
  input_x_bcast_.resize(RANK, kNoBroadcastValue);
  for (size_t i = 0; i < RANK; i++) {
    if (input_x_shape_[i] == input_shape_[i]) {
      continue;
    }
    if (input_x_shape_[i] == kNoBroadcastValue) {
      input_x_bcast_[i] = input_shape_[i];
    } else {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", broadcast not support, dim_x[" << std::to_string(i)
                        << "]=" << std::to_string(input_x_shape_[i]) << ", dim_y[" << std::to_string(i)
                        << "]=" << std::to_string(input_shape_[i]) << ".";
      return false;
    }
  }

  Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> input_x(static_cast<T *>(inputs[0]->addr), input_x_element_num);
  Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> output_y(static_cast<T *>(outputs[0]->addr),
                                                                 output_y_element_num);

  Eigen::DSizes<Eigen::DenseIndex, RANK> input_reshape;
  Eigen::DSizes<Eigen::DenseIndex, RANK> output_shape;
  Eigen::array<Eigen::DenseIndex, RANK> bcast;

  for (size_t i = 0; i < RANK; i++) {
    input_reshape[RANK - i - 1] = static_cast<Eigen::DenseIndex>(input_x_shape_[i]);
    output_shape[RANK - i - 1] = static_cast<Eigen::DenseIndex>(output_y_shape_[i]);
    bcast[RANK - i - 1] = static_cast<Eigen::DenseIndex>(input_x_bcast_[i]);
  }

  output_y.reshape(output_shape) = input_x.reshape(input_reshape).broadcast(bcast);
  return true;
}

std::vector<KernelAttr> ExpandCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> kernel_attr_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32)};

  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Expand, ExpandCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
