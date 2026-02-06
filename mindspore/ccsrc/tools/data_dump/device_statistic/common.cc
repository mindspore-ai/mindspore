/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "tools/data_dump/device_statistic/common.h"

#include <set>
#include <string>

#include "utils/log_adapter.h"

namespace mindspore {

namespace datadump {

const char KStatMax[] = "max";
const char KStatMin[] = "min";
const char KStatMean[] = "avg";
const char KStatL2Norm[] = "l2norm";
const char KCheckOverflow[] = "overflow";

void WarningOnce(const std::string &device_name, const std::string &type_name, const std::string &statistic_name) {
  static std::set<std::string> warning_once;
  std::string name = device_name + type_name + statistic_name;
  if (warning_once.find(name) != warning_once.end()) {
    return;
  } else {
    warning_once.insert(name);
    MS_LOG(WARNING) << "In the '" << device_name << "' platform, '" << type_name << "' is not supported for '"
                    << statistic_name << "' statistic dump.";
  }
}

}  // namespace datadump
}  // namespace mindspore
