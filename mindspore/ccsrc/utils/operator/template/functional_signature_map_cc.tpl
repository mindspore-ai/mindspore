/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "utils/operator/auto_generate/functional_signature_map.h"

#include <map>
#include <vector>
#include <string>

namespace mindspore::ops {
std::map<std::string, std::vector<std::string>> tensor_method_overload_signature_map = {
  ${tensor_method_sigs_map}
};

std::map<std::string, std::vector<std::string>> function_overload_signature_map = {
  ${mint_sigs_map}
};
}  // namespace mindspore::ops
