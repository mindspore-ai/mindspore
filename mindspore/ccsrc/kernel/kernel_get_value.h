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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNELGETVALUE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNELGETVALUE_H_

#include <optional>
#include <string>
#include <vector>
#include "include/backend/visible.h"
#include "ir/anf.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
BACKEND_EXPORT std::optional<std::vector<double>> TryGetFloatValueFromInputs(const std::vector<KernelTensorPtr> &inputs,
                                                                             const size_t input_index,
                                                                             const std::string &kernel_name,
                                                                             bool data_from_host);

inline bool TryGetFloatValue(const std::vector<KernelTensorPtr> &inputs, const size_t input_index,
                             const std::string &kernel_name, double *attr_value, bool data_from_host = true) {
  auto res = TryGetFloatValueFromInputs(inputs, input_index, kernel_name, data_from_host);
  if (!res.has_value()) {
    return false;
  }
  if (res.value().empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', value of the dynamic attr is empty!";
  }
  *attr_value = res.value()[0];
  return true;
}

inline bool TryGetFloatValue(const std::vector<KernelTensorPtr> &inputs, const size_t input_index,
                             const std::string &kernel_name, std::vector<double> *attr_value,
                             bool data_from_host = true) {
  auto res = TryGetFloatValueFromInputs(inputs, input_index, kernel_name, data_from_host);
  if (!res.has_value()) {
    return false;
  }
  *attr_value = res.value();
  return true;
}

BACKEND_EXPORT bool TryGetFloatValue(const CNodePtr &kernel_node, const size_t input_index,
                                     std::vector<double> *attr_value, bool data_from_host = true);

BACKEND_EXPORT std::optional<std::vector<int64_t>> TryGetIntValueFromInputs(const std::vector<KernelTensorPtr> &inputs,
                                                                            const size_t input_index,
                                                                            const std::string &kernel_name,
                                                                            bool data_from_host);

inline bool TryGetIntValue(const std::vector<KernelTensorPtr> &inputs, const size_t input_index,
                           const std::string &kernel_name, int64_t *attr_value, bool data_from_host = true) {
  auto res = TryGetIntValueFromInputs(inputs, input_index, kernel_name, data_from_host);
  if (!res.has_value()) {
    return false;
  }
  if (res.value().empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', value of the dynamic attr is empty!";
  }
  *attr_value = res.value()[0];
  return true;
}

inline bool TryGetIntValue(const std::vector<KernelTensorPtr> &inputs, const size_t input_index,
                           const std::string &kernel_name, std::vector<int64_t> *attr_value,
                           bool data_from_host = true) {
  auto res = TryGetIntValueFromInputs(inputs, input_index, kernel_name, data_from_host);
  if (!res.has_value()) {
    return false;
  }
  *attr_value = res.value();
  return true;
}

BACKEND_EXPORT bool TryGetIntValue(const CNodePtr &kernel_node, const size_t input_index,
                                   std::vector<int64_t> *attr_value, bool data_from_host = true);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNELGETVALUE_H_
