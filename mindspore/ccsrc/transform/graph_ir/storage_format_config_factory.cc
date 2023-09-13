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

#include "transform/graph_ir/storage_format_config_factory.h"

#include <utility>

namespace mindspore::transform {
StorageFormatConfig &StorageFormatConfig::set_index_format(size_t index, const std::string &format,
                                                           const DimsCheckFunc &func_, const std::string &expend_dims) {
  StorageFormatInfo info;
  info.format_ = format;
  info.expand_dims_ = expend_dims;
  info.func_ = func_;
  auto ret = storage_format_infoes_.emplace(index + 1, info);
  if (!ret.second) {
    MS_LOG(ERROR) << "Set index format op type: " << op_type_ << ", index: " << index << ", format: " << format
                  << ", expand_dims: " << expend_dims << " failed.";
  }
  return *this;
}

std::optional<StorageFormatInfo> StorageFormatConfig::GetStorageFormatInfo(size_t index) {
  auto iter = storage_format_infoes_.find(index);
  if (iter == storage_format_infoes_.end()) {
    return std::nullopt;
  }
  return iter->second;
}

StorageFormatConfigRegister &StorageFormatConfigRegister::GetInstance() {
  static StorageFormatConfigRegister inst;
  return inst;
}

StorageFormatConfig &StorageFormatConfigRegister::Register(const std::string &op_type) {
  auto iter = storage_format_configs_.find(op_type);
  if (iter != storage_format_configs_.end()) {
    return iter->second;
  }
  auto ret = storage_format_configs_.emplace(op_type, StorageFormatConfig(op_type));
  if (!ret.second) {
    MS_LOG(ERROR) << "Reg op failed: " << op_type;
  }
  return ret.first->second;
}

std::optional<StorageFormatConfig> StorageFormatConfigRegister::GetStorageFormatConfig(
  const std::string &op_type) const {
  auto iter = storage_format_configs_.find(op_type);
  if (iter == storage_format_configs_.end()) {
    return std::nullopt;
  }
  return iter->second;
}
}  // namespace mindspore::transform
