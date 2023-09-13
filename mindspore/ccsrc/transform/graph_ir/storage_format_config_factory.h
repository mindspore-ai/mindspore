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
#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_STROAGE_FORMAT_CONFIG_FACTORY_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_STROAGE_FORMAT_CONFIG_FACTORY_H_
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <map>

#include "include/transform/graph_ir/types.h"

namespace mindspore::transform {
using DimsCheckFunc = std::function<bool(const std::shared_ptr<GeTensorDesc> &)>;
struct StorageFormatInfo {
  std::string format_;
  std::string expand_dims_;
  DimsCheckFunc func_;
};

class StorageFormatConfig {
 public:
  explicit StorageFormatConfig(std::string op_type) : op_type_(std::move(op_type)) {}
  ~StorageFormatConfig() = default;
  StorageFormatConfig &set_index_format(size_t index, const std::string &format, const DimsCheckFunc &func_,
                                        const std::string &expend_dims = "");
  std::optional<StorageFormatInfo> GetStorageFormatInfo(size_t index);

 private:
  std::string op_type_;
  std::map<size_t, StorageFormatInfo> storage_format_infoes_{};
};

class StorageFormatConfigRegister {
 public:
  static StorageFormatConfigRegister &GetInstance();
  StorageFormatConfig &Register(const std::string &op_type);
  [[nodiscard]] std::optional<StorageFormatConfig> GetStorageFormatConfig(const std::string &op_type) const;

 private:
  StorageFormatConfigRegister() = default;
  ~StorageFormatConfigRegister() = default;
  std::map<std::string, StorageFormatConfig> storage_format_configs_;
};

#define REGISTER_STORAGE_FORMAT_CONFIG_IMPL(ctr, name)             \
  static transform::StorageFormatConfig &register_acl##name##ctr = \
    StorageFormatConfigRegister::GetInstance().Register(#name)

#define REGISTER_STORAGE_FORMAT_CONFIG(name) REGISTER_STORAGE_FORMAT_CONFIG_IMPL(__COUNTER__, name)
}  // namespace mindspore::transform

#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_STROAGE_FORMAT_CONFIG_FACTORY_H_
