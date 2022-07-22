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
namespace {
const std::vector<KernelAttr> kernel_attr = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64)},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128)},
};

constexpr size_t kInt8Size = 8;
constexpr size_t kInt16Size = 16;
constexpr size_t kInt32Size = 32;
constexpr size_t kInt64Size = 64;
constexpr size_t kInt128Size = 128;
const std::map<TypeId, size_t> data_size_map = {
  {kNumberTypeBool, kInt8Size},       {kNumberTypeUInt8, kInt8Size},        {kNumberTypeUInt16, kInt16Size},
  {kNumberTypeUInt32, kInt32Size},    {kNumberTypeUInt64, kInt64Size},      {kNumberTypeInt8, kInt8Size},
  {kNumberTypeInt16, kInt16Size},     {kNumberTypeInt32, kInt32Size},       {kNumberTypeInt64, kInt64Size},
  {kNumberTypeFloat16, kInt16Size},   {kNumberTypeFloat32, kInt32Size},     {kNumberTypeFloat64, kInt64Size},
  {kNumberTypeComplex64, kInt64Size}, {kNumberTypeComplex128, kInt128Size},
};
}  // namespace

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
    return false;
  }

  auto input_type_id = inputs[0]->GetDtype();
  if (data_size_map.find(input_type_id) != data_size_map.end()) {
    data_size_ = data_size_map.at(input_type_id);
  } else {
    MS_LOG(ERROR) << "'" << kernel_name_
                  << "' does not supported data type, the dtype of input must be bool, uint8, uint16, uint32, uint64, "
                     "int8, int16, int32, int64, float16, float32, float64, complex64, complex128";
    return false;
  }
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
    auto ret = LaunchKernel(inputs[i], outputs[i]);
    if (!ret) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', calculate output " << i << " failed.";
      return false;
    }
    shape_info_.input_shape_[input_index] = 1;
  }
  return true;
}

bool MeshgridCpuKernelMod::LaunchKernel(const kernel::AddressPtr input, const kernel::AddressPtr output) {
  MS_ERROR_IF_NULL_W_RET_VAL(input->addr, false);
  MS_ERROR_IF_NULL_W_RET_VAL(output->addr, false);
  int status = static_cast<int>(NNACL_OK);
  switch (data_size_) {
    case kInt8Size:
      status = BroadcastToSize8(input->addr, &shape_info_, output->addr);
      break;
    case kInt16Size:
      status = BroadcastToSize16(input->addr, &shape_info_, output->addr);
      break;
    case kInt32Size:
      status = BroadcastToSize32(input->addr, &shape_info_, output->addr);
      break;
    case kInt64Size:
      status = BroadcastToSize64(input->addr, &shape_info_, output->addr);
      break;
    case kInt128Size:
      status = BroadcastToSize128(input->addr, &shape_info_, output->addr);
      break;
    default:
      MS_LOG(ERROR)
        << "'" << kernel_name_
        << "' does not supported data type, the dtype of input must be bool, uint8, uint16, uint32, uint64, "
           "int8, int16, int32, int64, float16, float32, float64, complex64, complex128";
      return false;
  }
  if (status != static_cast<int>(NNACL_OK)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', broadcast input to output failed. Error code: " << status;
    return false;
  }
  return true;
}

std::vector<KernelAttr> MeshgridCpuKernelMod::GetOpSupport() { return kernel_attr; }

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Meshgrid, MeshgridCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
