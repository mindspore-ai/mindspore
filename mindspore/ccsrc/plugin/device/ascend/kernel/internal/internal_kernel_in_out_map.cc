/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include <stdarg.h>
#include <algorithm>
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/acme/acme_helper.h"

namespace mindspore {
namespace kernel {
InternalKernelModInOutMap *InternalKernelModInOutMap::GetInstance() {
  static InternalKernelModInOutMap instance;
  return &instance;
}

void InternalKernelModInOutMap::AppendKernelMap(const std::string &op_name, InternalKernelMapDtype map_dtype,
                                                std::vector<int> idx) {
  if (map_dtype == INTERNEL_KERNEL_MAP_INPUT) {
    input_idx_[op_name] = idx;
  } else if (map_dtype == INTERNEL_KERNEL_MAP_OUTPUT) {
    output_idx_[op_name] = idx;
  }
}

void InternalKernelModInOutMap::AppendMutableList(const std::string &op_name, InternalKernelMapDtype map_dtype) {
  if (map_dtype == INTERNEL_KERNEL_MAP_INPUT) {
    mutable_input_list_.insert(op_name);
  } else if (map_dtype == INTERNEL_KERNEL_MAP_OUTPUT) {
    mutable_output_list_.insert(op_name);
  }
}

std::vector<int> InternalKernelModInOutMap::GetKernelInMap(const std::string &op_name, bool *is_mutable) {
  if (is_mutable == nullptr) {
    return {};
  }

  auto map_iter = input_idx_.find(op_name);
  if (map_iter != input_idx_.end()) {
    *is_mutable = false;
    return map_iter->second;
  }

  *is_mutable = std::find(mutable_input_list_.begin(), mutable_input_list_.end(), op_name) != mutable_input_list_.end();
  return {};
}

std::vector<int> InternalKernelModInOutMap::GetKernelOutMap(const std::string &op_name, bool *is_mutable) {
  if (is_mutable == nullptr) {
    return {};
  }

  auto map_iter = output_idx_.find(op_name);
  if (map_iter != output_idx_.end()) {
    *is_mutable = false;
    return map_iter->second;
  }

  *is_mutable =
    std::find(mutable_output_list_.begin(), mutable_output_list_.end(), op_name) != mutable_output_list_.end();
  return {};
}

std::vector<int64_t> InternalKernelModInOutMap::MapInternelInputDtypes(const std::string &op_name,
                                                                       const std::vector<TypeId> &ms_dtypes) {
  std::vector<int64_t> internel_dtypes;
  auto map_iter = input_idx_.find(op_name);
  if (map_iter == input_idx_.end()) {
    return internel_dtypes;
  }
  auto idx_list = map_iter->second;
  for (size_t i = 0; i < idx_list.size(); i++) {
    internel_dtypes.push_back(InternalKernelUtils::ToInternalDType(ms_dtypes[idx_list.at(i)]));
  }
  return internel_dtypes;
}

std::vector<int64_t> InternalKernelModInOutMap::MapInternelOutputDtypes(const std::string &op_name,
                                                                        const std::vector<TypeId> &ms_dtypes) {
  std::vector<int64_t> internel_dtypes;
  auto map_iter = output_idx_.find(op_name);
  if (map_iter == output_idx_.end()) {
    return internel_dtypes;
  }
  auto idx_list = map_iter->second;
  for (size_t i = 0; i < idx_list.size(); i++) {
    internel_dtypes.push_back(InternalKernelUtils::ToInternalDType(ms_dtypes[idx_list.at(i)]));
  }
  return internel_dtypes;
}

std::vector<acme::DataType> InternalKernelModInOutMap::MapAcmeInputDtypes(const std::string &op_name,
                                                                          const std::vector<TypeId> &ms_dtypes) {
  std::vector<acme::DataType> acme_dtypes;
  auto map_iter = input_idx_.find(op_name);
  if (map_iter == input_idx_.end()) {
    return acme_dtypes;
  }
  auto idx_list = map_iter->second;
  for (size_t i = 0; i < idx_list.size(); i++) {
    acme_dtypes.push_back(TransAcmeDataType(ms_dtypes[idx_list.at(i)]));
  }
  return acme_dtypes;
}

std::vector<acme::DataType> InternalKernelModInOutMap::MapAcmeOutputDtypes(const std::string &op_name,
                                                                           const std::vector<TypeId> &ms_dtypes) {
  std::vector<acme::DataType> acme_dtypes;
  auto map_iter = output_idx_.find(op_name);
  if (map_iter == output_idx_.end()) {
    return acme_dtypes;
  }
  auto idx_list = map_iter->second;
  for (size_t i = 0; i < idx_list.size(); i++) {
    acme_dtypes.push_back(TransAcmeDataType(ms_dtypes[idx_list.at(i)]));
  }
  return acme_dtypes;
}

InternalKernelModInOutRegistrar::InternalKernelModInOutRegistrar(const std::string op_name, const int map_type,
                                                                 int total_count, ...) {
  if (total_count == INTERNEL_KERNEL_IN_OUT_MUTABLE_LENGTH) {
    InternalKernelModInOutMap::GetInstance()->AppendMutableList(op_name, (InternalKernelMapDtype)map_type);
    return;
  }

  std::vector<int> idx_list;
  va_list ptr;
  va_start(ptr, total_count);
  for (int i = 0; i < total_count; i++) {
    idx_list.push_back(va_arg(ptr, int));
  }
  va_end(ptr);
  InternalKernelModInOutMap::GetInstance()->AppendKernelMap(op_name, (InternalKernelMapDtype)map_type, idx_list);
}
}  // namespace kernel
}  // namespace mindspore
