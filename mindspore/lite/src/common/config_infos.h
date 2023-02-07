/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_COMMON_CONFIG_INFOS_H_
#define MINDSPORE_LITE_SRC_COMMON_CONFIG_INFOS_H_

#include <string>
#include <map>
#include <vector>
#include "include/api/visible.h"
#include "mindapi/base/shape_vector.h"

namespace mindspore {
using ConfigInfos = std::map<std::string, std::map<std::string, std::string>>;

struct ProfileInputInfo {
  std::string name;
  bool is_dynamic_shape = false;
  ShapeVector input_shape;
};

struct ProfileInputRange {
  ShapeVector min_dims;
  ShapeVector opt_dims;
  ShapeVector max_dims;
};

struct ProfileItem {
  std::vector<ProfileInputRange> inputs;
};

struct ProfileConfigs {
  std::vector<ProfileInputInfo> input_infos;
  std::vector<ProfileItem> profiles;
};

class MS_API ProfileParser {
 public:
  static bool Parse(const std::map<std::string, std::string> &context, bool require_opt_when_dym,
                    ProfileConfigs *profile_configs);

  static bool ReorderByInputNames(const std::vector<std::string> &input_names, ProfileConfigs *profile_configs);

  static std::string GetOption(const std::map<std::string, std::string> &context, const std::string &option,
                               const std::string &default_val = "");

 private:
  static bool ParseOptDims(const std::string &opt_dims_str, ProfileConfigs *profile_configs);
  static bool ParseDynamicDims(const std::string &dynamic_dims_str, ProfileConfigs *profile_configs);
  static bool ParseInputShape(const std::string &input_shapes_str, ProfileConfigs *profile_configs);

  static bool ParseRangeStr(const std::string &range_str, int64_t *min_ptr, int64_t *max_ptr);
  static bool ParseOptDimStr(const std::string &opt_dim_str, int64_t *opt_ptr);
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_CONFIG_INFOS_H_
