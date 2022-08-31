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

#ifndef MINDSPORE_LITE_SRC_COMMON_CONFIG_FILE_H_
#define MINDSPORE_LITE_SRC_COMMON_CONFIG_FILE_H_

#include <limits.h>
#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include "src/common/utils.h"
#include "src/common/log_adapter.h"
#include "src/common/config_infos.h"
#include "ir/dtype/type_id.h"

namespace mindspore {
namespace lite {
constexpr int MAX_CONFIG_FILE_LENGTH = 1024;

int GetAllSectionInfoFromConfigFile(const std::string &file, ConfigInfos *config);

void ParserExecutionPlan(const std::map<std::string, std::string> *config_infos,
                         std::map<std::string, TypeId> *data_type_plan);

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_CONFIG_FILE_H_
