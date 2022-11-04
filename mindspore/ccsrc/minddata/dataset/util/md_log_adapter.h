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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_MD_LOG_ADAPTER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_MD_LOG_ADAPTER_H_

#include <iostream>
#include <string>
#include <utility>

#include "include/api/status.h"

namespace mindspore {
namespace dataset {
class MDLogAdapter {
 public:
  MDLogAdapter() = default;

  ~MDLogAdapter() = default;

  static Status Apply(Status *rc);

  static std::string ConstructMsg(const enum StatusCode &status_code, const std::string &code_as_string,
                                  const std::string &status_msg, const int line_of_code, const std::string &file_name,
                                  const std::string &err_description);
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_MD_LOG_ADAPTER_H
