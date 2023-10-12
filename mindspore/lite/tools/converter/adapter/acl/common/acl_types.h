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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_COMMON_ACL_TYPES_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_COMMON_ACL_TYPES_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "include/api/data_type.h"
#include "include/api/context.h"

namespace mindspore {
namespace lite {
namespace acl {
struct AclModelOptionCfg {
  bool offline;
  int32_t device_id = 0;
  int32_t rank_id = 0;
  DataType output_type;
  std::vector<size_t> dynamic_batch_size;
  std::map<int32_t, std::vector<int32_t>> input_shape_map;
  std::string input_format;
  std::string input_shape;
  std::string precision_mode;
  std::string op_select_impl_mode;
  std::string fusion_switch_config_file_path;
  std::string buffer_optimize;
  std::string insert_op_config_file_path;
  std::string dynamic_image_size;
  std::string dynamic_dims;
  std::string om_file_path;
  std::string aoe_mode;
  std::string profiling_path;
  std::string dump_path;
  std::string dump_model_name;
  std::string custom_opp_path;
  std::vector<std::string> plugin_custom_ops;
  std::map<std::string, std::vector<std::string>> enable_custom_fusion_pattern;
  std::map<std::string, std::vector<std::string>> disable_custom_fusion_pattern;
  std::map<std::string, std::string> init_options_map;
  std::map<std::string, std::string> build_options_map;
  std::map<std::string, std::string> aoe_global_options_map;
  std::map<std::string, std::string> aoe_tuning_options_map;
};
}  // namespace acl
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_COMMON_ACL_TYPES_H_
