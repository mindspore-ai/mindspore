/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/kernel_get_value.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/framework_utils.h"
#include "kernel/kernel.h"
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;

namespace mindspore {
namespace kernel {
std::vector<double> GetFloatValueFromData(void *const data_c, const TypeId &type_id, size_t data_size,
                                          const size_t input_index, const std::string &kernel_name) {
  std::vector<double> tensor_value;
  MS_EXCEPTION_IF_NULL(data_c);
  if (type_id == kNumberTypeFloat32) {
    auto tensor_data = static_cast<float *>(data_c);
    MS_EXCEPTION_IF_NULL(tensor_data);
    tensor_value.assign(tensor_data, tensor_data + data_size / sizeof(float));
  } else if (type_id == kNumberTypeFloat64) {
    auto tensor_data = static_cast<double *>(data_c);
    MS_EXCEPTION_IF_NULL(tensor_data);
    tensor_value.assign(tensor_data, tensor_data + data_size / sizeof(double));
  } else {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name << "', the " << input_index
                            << "th input must be a Tensor[Float32] or Tensor[FLoat64] type, but got "
                            << TypeIdLabel(type_id);
  }
  return tensor_value;
}

std::optional<std::vector<double>> TryGetFloatValueFromInputs(const std::vector<KernelTensorPtr> &inputs,
                                                              const size_t input_index, const std::string &kernel_name,
                                                              bool data_from_host) {
  if (inputs.size() <= input_index) {
    MS_LOG(DEBUG) << "For '" << kernel_name << "', inputs size is " << inputs.size() << ", but require " << input_index;
    return std::nullopt;
  }

  AddressPtr data{nullptr};
  if (data_from_host) {
    data = inputs[input_index]->GetHostData();
  } else {
    data = inputs[input_index]->GetData();
  }

  // The value of dynamic attr can only be obtained after the InferOp() is executed.
  if (data == nullptr || data->addr == nullptr) {
    MS_LOG(DEBUG) << "For '" << kernel_name << "', fail to find the " << input_index << "th input's data.";
    return std::nullopt;
  }

  const auto &data_format = inputs[input_index]->GetFormat();
  if (data_format != mindspore::Format::DEFAULT_FORMAT && data_format != mindspore::Format::NCHW) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "',  the format of the " << input_index
                      << "th input currently should be the default format and does not support " << data_format;
  }

  return GetFloatValueFromData(data->addr, inputs[input_index]->GetDtype(), data->size, input_index, kernel_name);
}

bool TryGetFloatValue(const CNodePtr &kernel_node, const size_t input_index, std::vector<double> *attr_value,
                      bool data_from_host) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto args = GetArgsFromCNode(kernel_node);
  if (args == nullptr) {
    return false;
  }
  auto op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  auto res = TryGetFloatValueFromInputs(args->inputs, input_index, op_name, data_from_host);
  if (!res.has_value()) {
    return false;
  }
  *attr_value = res.value();
  return true;
}

std::vector<int64_t> GetIntValueFromData(void *const data_c, const TypeId &type_id, size_t data_size,
                                         const size_t input_index, const std::string &kernel_name) {
  std::vector<int64_t> tensor_value;
  MS_EXCEPTION_IF_NULL(data_c);
  if (type_id == kNumberTypeInt32) {
    auto tensor_data = reinterpret_cast<int32_t *>(data_c);
    MS_EXCEPTION_IF_NULL(tensor_data);
    tensor_value.assign(tensor_data, tensor_data + data_size / sizeof(int32_t));
  } else if (type_id == kNumberTypeInt64) {
    auto tensor_data = reinterpret_cast<int64_t *>(data_c);
    MS_EXCEPTION_IF_NULL(tensor_data);
    tensor_value.assign(tensor_data, tensor_data + data_size / sizeof(int64_t));
  } else {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name << "', the " << input_index
                            << "th input must be a Tensor[Int64] or Tensor[Int32] type, but got "
                            << TypeIdLabel(type_id);
  }
  return tensor_value;
}

std::optional<std::vector<int64_t>> TryGetIntValueFromInputs(const std::vector<KernelTensorPtr> &inputs,
                                                             const size_t input_index, const std::string &kernel_name,
                                                             bool data_from_host) {
  if (inputs.size() <= input_index) {
    MS_LOG(DEBUG) << "For '" << kernel_name << "', inputs size is " << inputs.size() << ", but require " << input_index;
    return std::nullopt;
  }

  AddressPtr data{nullptr};
  if (data_from_host) {
    data = inputs[input_index]->GetHostData();
  } else {
    data = inputs[input_index]->GetData();
  }

  // The value of dynamic attr can only be obtained after the InferOp() is executed.
  if (data == nullptr || data->addr == nullptr) {
    MS_LOG(DEBUG) << "For '" << kernel_name << "', fail to find the " << input_index << "th input's data.";
    return std::nullopt;
  }

  const auto &data_format = inputs[input_index]->GetFormat();
  if (data_format != mindspore::Format::DEFAULT_FORMAT && data_format != mindspore::Format::NCHW) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "',  the format of the " << input_index
                      << "th input currently should be the default format and does not support " << data_format;
  }

  return GetIntValueFromData(data->addr, inputs[input_index]->GetDtype(), data->size, input_index, kernel_name);
}

bool TryGetIntValue(const CNodePtr &kernel_node, const size_t input_index, std::vector<int64_t> *attr_value,
                    bool data_from_host) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto args = GetArgsFromCNode(kernel_node);
  if (args == nullptr) {
    return false;
  }
  auto op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  auto res = TryGetIntValueFromInputs(args->inputs, input_index, op_name, data_from_host);
  if (!res.has_value()) {
    return false;
  }
  *attr_value = res.value();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
