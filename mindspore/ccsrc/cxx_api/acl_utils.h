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

#ifndef MINDSPORE_CCSRC_CXX_API_ACL_UTILS_H
#define MINDSPORE_CCSRC_CXX_API_ACL_UTILS_H

#include <string>
#include <unordered_map>
#include "acl/acl_base.h"
namespace mindspore {
static inline bool IsAscend910Soc() {
  const char *soc_name_c = aclrtGetSocName();
  if (soc_name_c == nullptr) {
    return false;
  }
  std::string soc_name(soc_name_c);
  if (soc_name.find("910") == std::string::npos) {
    return false;
  }
  return true;
}

static inline bool IsAscendNo910Soc() {
  const char *soc_name_c = aclrtGetSocName();
  if (soc_name_c == nullptr) {
    return false;
  }
  std::string soc_name(soc_name_c);
  if (soc_name.find("910") != std::string::npos) {
    return false;
  }
  return true;
}

static inline std::string TransforPrecisionToAcl(const std::string &precision_mode) {
  static const std::unordered_map<std::string, std::string> precision_map = {
    {"enforce_fp32", "force_fp32"},
    {"preferred_fp32", "allow_fp32_to_fp16"},
    {"enforce_fp16", "force_fp16"},
    {"enforce_origin", "must_keep_origin_dtype"},
    {"preferred_optimal", "allow_mix_precision"}};
  return precision_map.find(precision_mode) != precision_map.end() ? precision_map.at(precision_mode) : precision_mode;
}
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_CXX_API_ACL_UTILS_H
