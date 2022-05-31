/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_ACL_OPTION_PARAM_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_ACL_OPTION_PARAM_PARSER_H_
#include <string>
#include "tools/converter/config_parser/config_file_parser.h"
#include "tools/converter/adapter/acl/common/acl_types.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
class AclOptionParamParser {
 public:
  STATUS ParseAclOptionCfg(const AclOptionCfgString &acl_option_string, acl::AclModelOptionCfg *acl_option_cfg);

 private:
  STATUS ParseCommon(const AclOptionCfgString &acl_option_string, acl::AclModelOptionCfg *acl_option_cfg);
  STATUS ParseDeviceId(const std::string &device_id, acl::AclModelOptionCfg *acl_option_cfg);
  STATUS ParseOutputType(const std::string &output_type, acl::AclModelOptionCfg *acl_option_cfg);
  STATUS ParseDynamicBatchSize(const std::string &dynamic_size, acl::AclModelOptionCfg *acl_option_cfg);
  STATUS ParseInputShapeVector(const std::string &input_shape_vector, acl::AclModelOptionCfg *acl_option_cfg);
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_ACL_OPTION_PARAM_PARSER_H_
