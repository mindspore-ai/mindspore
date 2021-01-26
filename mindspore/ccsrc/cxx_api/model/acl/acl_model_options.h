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

#ifndef MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_OPTION_PARSER_H
#define MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_OPTION_PARSER_H
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <memory>
#include "include/api/types.h"
#include "include/api/status.h"
#include "include/api/context.h"

namespace mindspore {
struct AclModelOptions {
  // build options
  std::string insert_op_cfg_path;
  std::string input_format;
  std::string input_shape;
  std::string output_type;
  std::string precision_mode;
  std::string op_select_impl_mode;
  std::string soc_version = "Ascend310";

  explicit AclModelOptions(const std::shared_ptr<Context> &context);
  ~AclModelOptions() = default;

  // return tuple<init_options, build_options>
  std::tuple<std::map<std::string, std::string>, std::map<std::string, std::string>> GenAclOptions() const;
  std::string GenAclOptionsKey() const;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_OPTION_PARSER_H
