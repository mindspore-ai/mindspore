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

#include "plugin/device/ascend/kernel/tbe/op_impl_mode_config.h"
#include <string>

#include "utils/ms_context.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kByOpType = "[ByOpType]";
constexpr auto kByNodeName = "[ByNodeName]";

std::string Trim(const std::string &input) {
  const char WHITESPACE[] = "\t\n\v\f\r ";
  auto begin = input.find_first_not_of(WHITESPACE);
  if (begin == std::string::npos) {
    return "";
  }
  auto end = input.find_last_not_of(WHITESPACE);
  return input.substr(begin, end - begin + 1);
}
}  // namespace

OpImplModeConfig &OpImplModeConfig::GetInstance() {
  static OpImplModeConfig instance{};
  return instance;
}

void OpImplModeConfig::Initialize() {
  op_name_impl_mode_map_.clear();
  op_type_impl_mode_map_.clear();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto op_precision_mode_config_path = ms_context->get_param<std::string>(MS_CTX_OP_PRECISION_MODE);
  if (op_precision_mode_config_path.empty()) {
    return;
  }

  auto config_real_path = FileUtils::GetRealPath(op_precision_mode_config_path.c_str());
  if (!config_real_path.has_value()) {
    MS_LOG(EXCEPTION) << "Get op_precision_mode config file real path failed with path: "
                      << op_precision_mode_config_path << ", please check context setting.";
  }
  GetOpPrecisionModeConfigFromFile(config_real_path.value());
}

void OpImplModeConfig::GetOpPrecisionModeConfigFromFile(const std::string &file_path) {
  MS_LOG(INFO) << "Load op_precision_mode config from " << file_path;
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    MS_LOG(WARNING) << "Cannot open op_precision_mode config file: " << file_path;
    return;
  }

  std::string line;
  bool by_op_type = true;
  while (std::getline(ifs, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    size_t equal_pos = line.find('=');
    if (equal_pos == std::string::npos) {
      std::string line_trim = Trim(line);
      if (line_trim == kByNodeName) {
        by_op_type = false;
      } else if (line_trim == kByOpType) {
        by_op_type = true;
      }
      continue;
    }
    ParseOneLine(line, by_op_type, equal_pos);
  }
  ifs.close();
}

void OpImplModeConfig::ParseOneLine(const std::string &line, bool by_op_type, size_t equal_pos) {
  auto op_type_or_name = line.substr(0, equal_pos);
  auto impl_mode = line.substr(equal_pos + 1);
  op_type_or_name = Trim(op_type_or_name);
  impl_mode = Trim(impl_mode);
  if (op_type_or_name.empty() || impl_mode.empty()) {
    return;
  }
  if (by_op_type) {
    (void)op_type_impl_mode_map_.emplace(op_type_or_name, impl_mode);
  } else {
    (void)op_name_impl_mode_map_.emplace(op_type_or_name, impl_mode);
  }
}

std::string OpImplModeConfig::GetOpImplMode(const std::string &op_name, const std::string &op_type) const {
  auto impl_mode = GetOpImplModeByName(op_name);
  if (impl_mode.empty()) {
    impl_mode = GetOpImplModeByType(op_type);
  }
  return impl_mode;
}

std::string OpImplModeConfig::GetOpImplModeByName(const std::string &op_name) const {
  auto iter = op_name_impl_mode_map_.find(op_name);
  if (iter == op_name_impl_mode_map_.end()) {
    return "";
  }
  return iter->second;
}

std::string OpImplModeConfig::GetOpImplModeByType(const std::string &op_type) const {
  auto iter = op_type_impl_mode_map_.find(op_type);
  if (iter == op_type_impl_mode_map_.end()) {
    return "";
  }
  return iter->second;
}
}  // namespace kernel
}  // namespace mindspore
