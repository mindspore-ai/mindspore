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
#include "include/api/types.h"
#include "include/api/status.h"

namespace mindspore::api {
struct AclModelOptions {
  std::string dump_cfg_path;
  std::string dvpp_cfg_path;
  std::string output_node;  // todo: at convert.cc::BuildGraph(), no atc options
  // build options
  std::string insert_op_cfg_path;
  std::string input_format;
  std::string input_shape;
  std::string dynamic_batch_size;
  std::string dynamic_image_size;
  std::string dynamic_dims;
  std::string serial_nodes_name;
  std::string output_type;

  explicit AclModelOptions(const std::map<std::string, std::string> &options);
  ~AclModelOptions() = default;

  std::map<std::string, std::string> GenAclOptions() const;
};
}  // namespace mindspore::api

#endif  // MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_OPTION_PARSER_H
