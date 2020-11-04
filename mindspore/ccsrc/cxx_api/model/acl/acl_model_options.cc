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
#include "external/ge/ge_api_types.h"

namespace mindspore::api {
static std::string ParseOption(const std::map<std::string, std::string> &options, const std::string &key) {
  auto iter = options.find(key);
  if (iter != options.end()) {
    return iter->second;
  }
  return "";
}

AclModelOptions::AclModelOptions(const std::map<std::string, std::string> &options) {
  dump_cfg_path = ParseOption(options, kModelOptionDumpCfgPath);
  dvpp_cfg_path = ParseOption(options, kModelOptionDvppCfgPath);
  output_node = ParseOption(options, kModelOptionOutputNode);
  // to acl
  insert_op_cfg_path = ParseOption(options, kModelOptionInsertOpCfgPath);
  input_format = ParseOption(options, kModelOptionInputFormat);
  input_shape = ParseOption(options, kModelOptionInputShape);
  dynamic_batch_size = ParseOption(options, kModelOptionInputShape);
  dynamic_image_size = ParseOption(options, kModelOptionInputShape);
  dynamic_dims = ParseOption(options, kModelOptionInputShape);
  serial_nodes_name = ParseOption(options, kModelOptionSerialInput);
  output_type = ParseOption(options, kModelOptionOutputType);
}

std::map<std::string, std::string> AclModelOptions::GenAclOptions() const {
  const std::map<std::string const *, std::string> acl_options_map = {
    {&insert_op_cfg_path, ge::ir_option::INSERT_OP_FILE},
    {&input_format, ge::ir_option::INPUT_FORMAT},
    {&input_shape, ge::ir_option::INPUT_SHAPE},
    {&dynamic_batch_size, ge::ir_option::DYNAMIC_BATCH_SIZE},
    {&dynamic_image_size, ge::ir_option::DYNAMIC_IMAGE_SIZE},
    {&dynamic_dims, ge::ir_option::DYNAMIC_DIMS},
    {&serial_nodes_name, ge::ir_option::INPUT_FP16_NODES},
    {&output_type, ge::ir_option::OUTPUT_TYPE},
  };

  std::map<std::string, std::string> acl_options;
  for (auto [ms_option, acl_option_key] : acl_options_map) {
    if (ms_option == nullptr || ms_option->empty()) {
      continue;
    }
    acl_options.emplace(acl_option_key, *ms_option);
  }
  return acl_options;
}

}  // namespace mindspore::api
