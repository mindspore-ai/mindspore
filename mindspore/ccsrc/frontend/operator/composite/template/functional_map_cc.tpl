/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "frontend/operator/composite/auto_generate/functional_map.h"

#include <map>
#include <set>
#include <vector>
#include <string>
#include "frontend/operator/composite/functional_overload.h"
${ops_inc}

namespace mindspore::ops {
${deprecated_method_decl}

std::map<std::string, std::vector<ValuePtr>> tensor_method_overload_map = {
  ${tensor_method_map}
};

std::map<std::string, std::vector<ValuePtr>> function_overload_map = {
  ${mint_map}
};

std::map<std::string, std::set<std::string>> tensor_method_kwonlyargs_map = {
  ${tensor_method_kwonlyargs_map}
};

std::map<std::string, std::set<std::string>> function_kwonlyargs_map = {
  ${mint_kwonlyargs_map}
};

std::map<std::string, size_t> tensor_method_varargs_map = {
  ${tensor_varargs_map}
};

std::map<std::string, size_t> function_varargs_map = {
  ${mint_varargs_map}
};
}  // namespace mindspore::ops
