/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/kernel/ascend/custom/kernel_mod_impl/custom_kernel_factory.h"
#include <algorithm>
#include <numeric>

namespace mindspore {
namespace kernel {

CustomKernelFactory &CustomKernelFactory::Instance() {
  static CustomKernelFactory instance;
  return instance;
}

bool CustomKernelFactory::Register(const std::string &op_name, const KernelCreator &creator) {
  return creators_.emplace(op_name, creator).second;
}

KernelModPtr CustomKernelFactory::Create(const std::string &op_name) {
  auto it = creators_.find(op_name);
  if (it != creators_.end()) {
    return (it->second)();
  }
  return nullptr;
}

bool CustomKernelFactory::IsRegistered(const std::string &op_name) {
  return creators_.find(op_name) != creators_.end();
}

bool CustomKernelFactory::RegisterHardwareFormatMapping(const std::string &op_name,
                                                        const HardwareFormatMapping &hardware_mapping) {
  auto it = hardware_format_mappings_.find(op_name);
  if (it == hardware_format_mappings_.end()) {
    hardware_format_mappings_[op_name] = {hardware_mapping};
    return true;
  } else {
    auto existing_mapping_it = std::find_if(it->second.begin(), it->second.end(),
                                            [&hardware_mapping](const HardwareFormatMapping &existing_mapping) {
                                              return existing_mapping.hardware == hardware_mapping.hardware;
                                            });
    if (existing_mapping_it != it->second.end()) {
      for (const auto &format_pair : hardware_mapping.format_mappings) {
        existing_mapping_it->format_mappings[format_pair.first] = format_pair.second;
      }
      return true;
    }
    it->second.push_back(hardware_mapping);
    return true;
  }
}

bool CustomKernelFactory::HasFormatMapping(const std::string &op_name) const {
  auto hardware_it = hardware_format_mappings_.find(op_name);
  return hardware_it != hardware_format_mappings_.end() && !hardware_it->second.empty();
}

const std::unordered_map<std::string, std::vector<HardwareFormatMapping>> &
CustomKernelFactory::GetAllHardwareFormatMappings() const {
  return hardware_format_mappings_;
}

bool CustomKernelFactory::FindMatchingFormatMapping(const std::string &op_name,
                                                    const std::vector<std::string> &input_formats,
                                                    const std::string &hardware, KernelFormatMapping *result) const {
  MS_EXCEPTION_IF_NULL(result);

  auto hardware_it = hardware_format_mappings_.find(op_name);
  if (hardware_it != hardware_format_mappings_.end()) {
    auto hardware_mapping_it = std::find_if(
      hardware_it->second.begin(), hardware_it->second.end(),
      [&hardware](const HardwareFormatMapping &hardware_mapping) { return hardware_mapping.hardware == hardware; });
    if (hardware_mapping_it != hardware_it->second.end()) {
      std::string key = std::accumulate(
        input_formats.begin(), input_formats.end(), std::string{},
        [](std::string acc, const std::string &format) { return acc.empty() ? format : acc + "," + format; });

      auto format_it = hardware_mapping_it->format_mappings.find(key);
      if (format_it != hardware_mapping_it->format_mappings.end()) {
        *result = format_it->second;
        return true;
      }
    }
  }
  *result = KernelFormatMapping();
  return false;
}

}  // namespace kernel
}  // namespace mindspore
