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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_VALIDATORS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_VALIDATORS_H_

#include <nlohmann/json.hpp>
#include <string>

#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// validator Parameter in json file
inline Status ValidateParamInJson(const nlohmann::json &json_obj, const std::string &param_name,
                                  const std::string &operator_name) {
  if (json_obj.find(param_name) == json_obj.end()) {
    std::string err_msg = "Failed to find key '" + param_name + "' in " + operator_name +
                          "' JSON file or input dict, check input content of deserialize().";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_VALIDATORS_H_
