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

#include "plugin/device/cpu/kernel/meshgrid_cpu_kernel.h"
#include <algorithm>
#include "mindspore/core/ops/meshgrid.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
std::vector<KernelAttr> MeshgridCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MeshgridFunc> &pair) { return pair.first; });
  return support_list;
}

bool MeshgridCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.size() != outputs.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be equal, but get " << inputs.size()
                  << " and " << outputs.size();
    return false;
  }
  if (inputs.size() <= 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input size must greater than 1, but get " << inputs.size();
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::Meshgrid>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', Cast Meshgrid ops failed!";
    return false;
  }
  auto indexing = kernel_ptr->get_indexing();
  if (indexing == "xy") {
    swap_indexing_ = true;
  } else if (indexing == "ij") {
    swap_indexing_ = false;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value of 'indexing' must be \"xy\" or \"ij\", but get "
                  << indexing;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;
  return true;
}

int MeshgridCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (input_size_list_.size() != output_size_list_.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be equal, but get "
                  << input_size_list_.size() << " and " << output_size_list_.size();
    return KRET_RESIZE_FAILED;
  }
  shape_info_.input_shape_size_ = SizeToInt(inputs.size());
  shape_info_.output_shape_size_ = SizeToInt(inputs.size());
  if (shape_info_.output_shape_size_ > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of output must be at most 8. But get and the dimension of target shape: "
                  << shape_info_.output_shape_size_;
    return KRET_RESIZE_FAILED;
  }

  input_shape_lists_.clear();
  std::vector<int64_t> out_shape;
  // The input tensor must be 1-D tensor.
  for (auto &input : inputs) {
    auto shape = input->GetShapeVector();
    if (shape.size() != 1) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', each input tensor shape size must be 1, but get " << shape.size();
      return KRET_RESIZE_FAILED;
    }
    input_shape_lists_.push_back(shape[0]);
    out_shape.push_back(shape[0]);
  }
  if (swap_indexing_) {
    std::swap(out_shape[0], out_shape[1]);
  }
  for (int i = 0; i < shape_info_.input_shape_size_; i++) {
    shape_info_.input_shape_[IntToSize(i)] = 1;
  }
  for (size_t i = 0; i < out_shape.size(); i++) {
    shape_info_.output_shape_[i] = LongToInt(out_shape[i]);
  }

  for (auto &output : outputs) {
    auto shape = output->GetShapeVector();
    if (shape != out_shape) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', each output tensor shape should be the combination of all input tensor shape. But get the "
                       "shape of all inputs tensor shape: "
                    << Vector2Str(out_shape) << ", and the shape of output: " << Vector2Str(shape);
      return KRET_RESIZE_FAILED;
    }
  }
  return KRET_OK;
}

bool MeshgridCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                  const std::vector<AddressPtr> &outputs) {
  for (size_t i = 0; i < outputs.size(); i++) {
    auto input_index = (i <= 1 && swap_indexing_ == true) ? 1 - i : i;
    shape_info_.input_shape_[input_index] = LongToInt(input_shape_lists_[i]);
    auto ret = kernel_func_(this, inputs[i], outputs[i]);
    if (!ret) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', calculate output " << i << " failed.";
      return false;
    }
    shape_info_.input_shape_[input_index] = 1;
  }
  return true;
}

template <typename T>
bool MeshgridCpuKernelMod::LaunchKernel(const kernel::AddressPtr input, const kernel::AddressPtr output) {
  MS_ERROR_IF_NULL_W_RET_VAL(input->addr, false);
  MS_ERROR_IF_NULL_W_RET_VAL(output->addr, false);
  int status = static_cast<int>(NNACL_OK);
  if constexpr (std::is_same_v<T, bool>) {
    status = BROADCAST_TO(bool, reinterpret_cast<T *>(input->addr), &shape_info_, reinterpret_cast<T *>(output->addr));
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    status = BROADCAST_TO(int8_t, reinterpret_cast<int8_t *>(input->addr), &shape_info_,
                          reinterpret_cast<int8_t *>(output->addr));
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    status = BROADCAST_TO(int16_t, reinterpret_cast<int16_t *>(input->addr), &shape_info_,
                          reinterpret_cast<int16_t *>(output->addr));
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    status = BROADCAST_TO(int32_t, reinterpret_cast<int32_t *>(input->addr), &shape_info_,
                          reinterpret_cast<int32_t *>(output->addr));
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    status = BROADCAST_TO(int64_t, reinterpret_cast<int64_t *>(input->addr), &shape_info_,
                          reinterpret_cast<int64_t *>(output->addr));
  } else if constexpr (std::is_same_v<T, int8_t>) {
    status = BROADCAST_TO(int8_t, reinterpret_cast<int8_t *>(input->addr), &shape_info_,
                          reinterpret_cast<int8_t *>(output->addr));
  } else if constexpr (std::is_same_v<T, int16_t>) {
    status = BROADCAST_TO(int16_t, reinterpret_cast<int16_t *>(input->addr), &shape_info_,
                          reinterpret_cast<int16_t *>(output->addr));
  } else if constexpr (std::is_same_v<T, int32_t>) {
    status = BROADCAST_TO(int32_t, reinterpret_cast<int32_t *>(input->addr), &shape_info_,
                          reinterpret_cast<int32_t *>(output->addr));
  } else if constexpr (std::is_same_v<T, int64_t>) {
    status = BROADCAST_TO(int64_t, reinterpret_cast<int64_t *>(input->addr), &shape_info_,
                          reinterpret_cast<int64_t *>(output->addr));
  } else if constexpr (std::is_same_v<T, float16>) {
    status = BROADCAST_TO(int16_t, reinterpret_cast<int16_t *>(input->addr), &shape_info_,
                          reinterpret_cast<int16_t *>(output->addr));
  } else if constexpr (std::is_same_v<T, float>) {
    status = BROADCAST_TO(int32_t, reinterpret_cast<int32_t *>(input->addr), &shape_info_,
                          reinterpret_cast<int32_t *>(output->addr));
  } else if constexpr (std::is_same_v<T, double>) {
    status = BROADCAST_TO(int64_t, reinterpret_cast<int64_t *>(input->addr), &shape_info_,
                          reinterpret_cast<int64_t *>(output->addr));
  } else {
    MS_LOG(ERROR) << "'" << kernel_name_
                  << "' does not supported data type, the dtype of input must be bool, uint8, uint16, uint32, uint64, "
                     "int8, int16, int32, int64, float16, float32 or float64.";
    return false;
  }
  if (status != static_cast<int>(NNACL_OK)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', broadcast input to output failed. Error code: " << status;
    return false;
  }
  return true;
}

std::vector<std::pair<KernelAttr, MeshgridCpuKernelMod::MeshgridFunc>> MeshgridCpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &MeshgridCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &MeshgridCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &MeshgridCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &MeshgridCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &MeshgridCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &MeshgridCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &MeshgridCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &MeshgridCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &MeshgridCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &MeshgridCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &MeshgridCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &MeshgridCpuKernelMod::LaunchKernel<double>},
};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Meshgrid, MeshgridCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
