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

#include "cxx_api/any_utils.h"

namespace mindspore {
namespace {
template <class T, typename U = std::remove_cv_t<std::remove_reference_t<T>>>
static U GetValue(const std::map<std::string, std::any> &any_map, const std::string &key) {
  static const U empty_result{};
  auto iter = any_map.find(key);
  if (iter == any_map.end()) {
    return empty_result;
  }
  const std::any &value = iter->second;
  if (value.type() != typeid(U)) {
    return empty_result;
  }

  return std::any_cast<U>(value);
}
}  // namespace

void SetAnyValue(std::any *any, bool value) {
  if (any != nullptr) {
    *any = value;
  }
}

void SetAnyValue(std::any *any, int value) {
  if (any != nullptr) {
    *any = value;
  }
}

void SetAnyValue(std::any *any, uint32_t value) {
  if (any != nullptr) {
    *any = value;
  }
}

void SetAnyValue(std::any *any, const std::string &value) {
  if (any != nullptr) {
    *any = value;
  }
}

void SetAnyValue(std::any *any, DataType value) {
  if (any != nullptr) {
    *any = value;
  }
}

void SetAnyValue(std::any *any, const std::map<int, std::vector<int>> &value) {
  if (any != nullptr) {
    *any = value;
  }
}

bool GetAnyValueBool(const std::map<std::string, std::any> &any_map, const std::string &name) {
  return GetValue<bool>(any_map, name);
}

int GetAnyValueI32(const std::map<std::string, std::any> &any_map, const std::string &name) {
  return GetValue<int>(any_map, name);
}

uint32_t GetAnyValueU32(const std::map<std::string, std::any> &any_map, const std::string &name) {
  return GetValue<uint32_t>(any_map, name);
}

DataType GetAnyValueDataType(const std::map<std::string, std::any> &any_map, const std::string &name) {
  return GetValue<DataType>(any_map, name);
}

std::string GetAnyValueStr(const std::map<std::string, std::any> &any_map, const std::string &name) {
  return GetValue<std::string>(any_map, name);
}

std::map<int, std::vector<int>> GetAnyValueInputShape(const std::map<std::string, std::any> &any_map,
                                                      const std::string &name) {
  return GetValue<std::map<int, std::vector<int>>>(any_map, name);
}
}  // namespace mindspore
