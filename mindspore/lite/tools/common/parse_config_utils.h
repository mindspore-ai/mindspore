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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_PARSE_CONFIG_UTILS_H_
#define MINDSPORE_LITE_TOOLS_COMMON_PARSE_CONFIG_UTILS_H_
#include <string>
#include <map>
namespace mindspore {
namespace lite {
// the maps key is [section], and the value_map is {name,value}
int ParseConfigFile(const std::string &config_file_path,
                    std::map<std::string, std::map<std::string, std::string>> *maps,
                    std::map<int, std::map<std::string, std::string>> *model_param_infos);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_COMMON_PARSE_CONFIG_UTILS_H_
