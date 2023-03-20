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
#ifndef MINDSPORE_CCSRC_CXX_API_ANY_UTILS_H_
#define MINDSPORE_CCSRC_CXX_API_ANY_UTILS_H_

#include <cstdint>
#include <any>
#include <vector>
#include <string>
#include <map>
#include "include/api/visible.h"
#include "include/api/data_type.h"

namespace mindspore {
// std::any is not support to access across shared libraries, so add an adapter to access std::any
MS_API void SetAnyValue(std::any *any, bool value);
MS_API void SetAnyValue(std::any *any, int value);
MS_API void SetAnyValue(std::any *any, uint32_t value);
MS_API void SetAnyValue(std::any *any, const std::string &value);
MS_API void SetAnyValue(std::any *any, DataType value);
MS_API void SetAnyValue(std::any *any, const std::map<int, std::vector<int>> &value);

MS_API bool GetAnyValueBool(const std::map<std::string, std::any> &any_map, const std::string &name);
MS_API int GetAnyValueI32(const std::map<std::string, std::any> &any_map, const std::string &name);
MS_API uint32_t GetAnyValueU32(const std::map<std::string, std::any> &any_map, const std::string &name);
MS_API DataType GetAnyValueDataType(const std::map<std::string, std::any> &any_map, const std::string &name);
MS_API std::string GetAnyValueStr(const std::map<std::string, std::any> &any_map, const std::string &name);
MS_API std::map<int, std::vector<int>> GetAnyValueInputShape(const std::map<std::string, std::any> &any_map,
                                                             const std::string &name);
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_ANY_UTILS_H_
