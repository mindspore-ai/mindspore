/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "cxx_api/model/acl/acl_model_options.h"
#include <memory>
#include "utils/log_adapter.h"
#include "external/ge/ge_api_types.h"

namespace mindspore {
static const std::map<enum DataType, std::string> kSupportedDtypeOptionMap = {{DataType::kNumberTypeFloat16, "FP16"},
                                                                              {DataType::kNumberTypeFloat32, "FP32"},
                                                                              {DataType::kNumberTypeUInt8, "UINT8"}};

AclModelOptions::AclModelOptions(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    return;
  }
  insert_op_cfg_path = ModelContext::GetInsertOpConfigPath(context);
  input_format = ModelContext::GetInputFormat(context);
  input_shape = ModelContext::GetInputShape(context);

  auto out_type = ModelContext::GetOutputType(context);
  auto iter = kSupportedDtypeOptionMap.find(out_type);
  if (out_type == DataType::kTypeUnknown) {
    // do nothing
  } else if (iter == kSupportedDtypeOptionMap.end()) {
    MS_LOG(WARNING) << "Unsupported output type " << out_type << ", use FP32 as default.";
  } else {
    output_type = iter->second;
  }

  precision_mode = ModelContext::GetPrecisionMode(context);
  op_select_impl_mode = ModelContext::GetOpSelectImplMode(context);
}

std::tuple<std::map<std::string, std::string>, std::map<std::string, std::string>> AclModelOptions::GenAclOptions()
  const {
  const std::map<std::string const *, std::string> init_options_map = {
    {&op_select_impl_mode, ge::ir_option::OP_SELECT_IMPL_MODE},
    {&soc_version, ge::ir_option::SOC_VERSION},
  };

  const std::map<std::string const *, std::string> build_options_map = {
    {&insert_op_cfg_path, ge::ir_option::INSERT_OP_FILE}, {&input_format, ge::ir_option::INPUT_FORMAT},
    {&input_shape, ge::ir_option::INPUT_SHAPE},           {&output_type, ge::ir_option::OUTPUT_TYPE},
    {&precision_mode, ge::ir_option::PRECISION_MODE},
  };

  std::map<std::string, std::string> init_options;
  std::map<std::string, std::string> build_options;
  for (auto [ms_option, acl_option_key] : init_options_map) {
    if (ms_option == nullptr || ms_option->empty()) {
      continue;
    }
    MS_LOG(INFO) << "Option " << acl_option_key << " : " << *ms_option;
    init_options.emplace(acl_option_key, *ms_option);
  }

  for (auto [ms_option, acl_option_key] : build_options_map) {
    if (ms_option == nullptr || ms_option->empty()) {
      continue;
    }
    MS_LOG(INFO) << "Option " << acl_option_key << " : " << *ms_option;
    build_options.emplace(acl_option_key, *ms_option);
  }
  return {init_options, build_options};
}

std::string AclModelOptions::GenAclOptionsKey() const {
  auto [init_options, build_options] = GenAclOptions();
  std::string key_str;
  for (auto &[key, value] : init_options) {
    key_str += key + "^" + value + "^^";
  }
  for (auto &[key, value] : build_options) {
    key_str += key + "^" + value + "^^";
  }
  return key_str;
}
}  // namespace mindspore
