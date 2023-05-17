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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_OP_IMPL_MODE_CONFIG_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_OP_IMPL_MODE_CONFIG_
#include <string>
#include "utils/hash_map.h"

namespace mindspore {
namespace kernel {
// Parse op_precision_mode config file to set op_impl_mode
// config file example:
// [ByOpType]
// optype1=high_precision
// optype2=high_performance
// optype3=support_of_bound_index
//
// [ByNodeName]
// nodename1=high_precision
// nodename2=high_performance
// nodename3=support_of_bound_index
class OpImplModeConfig {
 public:
  static OpImplModeConfig &GetInstance();
  OpImplModeConfig(const OpImplModeConfig &) = delete;
  OpImplModeConfig &operator=(const OpImplModeConfig &) = delete;

  void Initialize();
  std::string GetOpImplMode(const std::string &op_name, const std::string &op_type) const;

 private:
  OpImplModeConfig() = default;
  ~OpImplModeConfig() {}

  void GetOpPrecisionModeConfigFromFile(const std::string &file_path);
  void ParseOneLine(const std::string &line, bool by_op_type, size_t equal_pos);
  std::string GetOpImplModeByName(const std::string &op_name) const;
  std::string GetOpImplModeByType(const std::string &op_type) const;

  mindspore::HashMap<std::string, std::string> op_name_impl_mode_map_;
  mindspore::HashMap<std::string, std::string> op_type_impl_mode_map_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_OP_IMPL_MODE_CONFIG_
