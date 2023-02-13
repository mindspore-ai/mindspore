/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/debug/print_gpu_kernel.h"
#include <tuple>
#include <functional>
#include <algorithm>
#include <memory>
#include "ir/tensor.h"
#include "ops/print.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "kernel/common_utils.h"

using mindspore::tensor::Tensor;

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

bool PrintGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  VARIABLE_NOT_USED(workspace);
  std::vector<void *> input_device_data;
  InitDeviceData(inputs, &input_device_data);
  int *output_address = GetDeviceAddress<int>(outputs, 0);
  // host initialization in byte for storage
  std::unique_ptr<uint8_t[]> input_host_data;
  int64_t sum_of_bytes = 0;
  for (size_t i = 0; i < input_info_.size(); i++) {
    sum_of_bytes += std::get<0>(input_info_[i]);
  }
  input_host_data = std::make_unique<uint8_t[]>(sum_of_bytes);
  // print core function
  size_t string_idx = 0;
  auto offset = input_host_data.get();
  for (size_t i = 0; i < input_flag_.size(); i++) {
    if (input_flag_[i] == -1) {
      std::cout << string_value_[string_idx] << std::endl;
      string_idx++;
    } else {
      size_t tensor_idx = LongToSize(input_flag_[i]);
      size_t size_to_move = std::get<0>(input_info_[tensor_idx]);
      std::string error_msg = "cudaMemcpyAsync print loop failed at input_device_data[";
      error_msg.append(std::to_string(tensor_idx));
      error_msg.append("].");
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(offset, input_device_data[tensor_idx], size_to_move, cudaMemcpyDeviceToHost,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        error_msg);
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(), "cudaDeviceSyncFailed - Print");
      auto current_string = GetString(tensor_idx, i, offset);
      std::cout << current_string << std::endl;
      offset += size_to_move;
    }
  }
  int output = 1;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(output_address, &output, sizeof(int), cudaMemcpyHostToDevice,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "cudaMemcpyAsync output failed");
  return true;
}

bool PrintGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Print>(base_operator);
  if (kernel_ptr->HasAttr("string_pos")) {
    string_value_ = kernel_ptr->get_string_value();
    string_pos_ = kernel_ptr->get_string_pos();
    auto value_type = kernel_ptr->get_value_type();
    auto value_type_pos = kernel_ptr->get_value_type_pos();
    for (size_t i = 0; i < value_type.size(); i++) {
      value_type_[value_type_pos[i]] = value_type[i];
    }
  }

  return true;
}

int PrintGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }
  size_t input_tensor_num = inputs.size();
  input_flag_ = SetInputFlag(&string_pos_, input_tensor_num);
  input_shape_.clear();
  input_info_.clear();
  for (size_t i = 0; i < input_tensor_num; i++) {
    auto input_shape = inputs[i]->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      return KRET_OK;
    }
    auto type_id = inputs[i]->GetDtype();
    size_t unit_size = UnitSizeInBytes(type_id);
    auto size_in_byte = std::accumulate(input_shape.begin(), input_shape.end(), unit_size, std::multiplies<size_t>());
    input_info_.push_back(std::make_tuple(size_in_byte, type_id));
    input_shape_.push_back(input_shape);
  }

  return ret;
}

void PrintGpuKernelMod::InitDeviceData(const std::vector<AddressPtr> &inputs, std::vector<void *> *input_device_data) {
  MS_EXCEPTION_IF_NULL(input_device_data);
  for (size_t i = 0; i < inputs.size(); i++) {
    TypeId type_id = std::get<1>(input_info_[i]);
    switch (type_id) {
      case kNumberTypeBool:
        input_device_data->push_back(GetDeviceAddress<bool>(inputs, i));
        break;
      case kNumberTypeInt8:
        input_device_data->push_back(GetDeviceAddress<int8_t>(inputs, i));
        break;
      case kNumberTypeInt16:
        input_device_data->push_back(GetDeviceAddress<int16_t>(inputs, i));
        break;
      case kNumberTypeInt32:
        input_device_data->push_back(GetDeviceAddress<int32_t>(inputs, i));
        break;
      case kNumberTypeInt64:
        input_device_data->push_back(GetDeviceAddress<int64_t>(inputs, i));
        break;
      case kNumberTypeUInt8:
        input_device_data->push_back(GetDeviceAddress<uint8_t>(inputs, i));
        break;
      case kNumberTypeUInt16:
        input_device_data->push_back(GetDeviceAddress<uint16_t>(inputs, i));
        break;
      case kNumberTypeUInt32:
        input_device_data->push_back(GetDeviceAddress<uint32_t>(inputs, i));
        break;
      case kNumberTypeUInt64:
        input_device_data->push_back(GetDeviceAddress<uint64_t>(inputs, i));
        break;
      case kNumberTypeFloat16:
        input_device_data->push_back(GetDeviceAddress<half>(inputs, i));
        break;
      case kNumberTypeFloat32:
        input_device_data->push_back(GetDeviceAddress<float>(inputs, i));
        break;
      case kNumberTypeFloat64:
        input_device_data->push_back(GetDeviceAddress<double>(inputs, i));
        break;
      case kNumberTypeComplex64:
        input_device_data->push_back(GetDeviceAddress<Complex<float>>(inputs, i));
        break;
      case kNumberTypeComplex128:
        input_device_data->push_back(GetDeviceAddress<Complex<double>>(inputs, i));
        break;
      default:
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the typeid cannot be " << type_id;
    }
  }
}

std::vector<int64_t> PrintGpuKernelMod::SetInputFlag(std::vector<int64_t> *string_pos, size_t input_tensor_num) {
  // -1 -> string position
  // others -> input tensor position
  std::vector<int64_t> res(string_pos->size() + input_tensor_num);
  // without string inputs
  int64_t value = 0;
  if (res.size() == input_tensor_num) {
    std::generate(res.begin(), res.end(), [&value]() { return value++; });
    return res;
  }
  for (size_t i = 0; i < string_pos->size(); i++) {
    if ((*string_pos)[i] < 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', string_pos cannot be a negative value";
    }
    auto index = IntToSize((*string_pos)[i]);
    res[index] = -1;
  }
  for (size_t i = 0; i < res.size(); i++) {
    if (res[i] != -1) {
      res[i] += value;
      value++;
    }
  }
  return res;
}

std::string PrintGpuKernelMod::GetString(size_t tensor_index, size_t original_index, void *input_host_data) {
  ShapeVector shape;
  size_t size_in_byte = std::get<0>(input_info_[tensor_index]);
  TypeId type_id = std::get<1>(input_info_[tensor_index]);
  (void)std::transform(input_shape_[tensor_index].begin(), input_shape_[tensor_index].end(), std::back_inserter(shape),
                       [](const size_t &value) { return static_cast<int64_t>(value); });
  Tensor current_tensor(type_id, shape, input_host_data, size_in_byte);
  if (value_type_.count(original_index) > 0) {
    // not a tensor
    auto out = current_tensor.data().ToString(type_id, shape, true);
    if (value_type_[original_index] != 0) {
      // tuple, not scalar
      (void)std::replace(out.begin(), out.end(), '[', '(');
      (void)std::replace(out.begin(), out.end(), ']', ')');
    }
    return out;
  }
  return current_tensor.ToStringNoLimit();
}

std::vector<KernelAttr> PrintGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Print, PrintGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
