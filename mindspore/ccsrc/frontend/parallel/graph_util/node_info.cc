/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/graph_util/node_info.h"

#include <string>

#include "ir/anf.h"
#include "ir/param_value.h"
#include "pipeline/jit/parse/python_adapter.h"

namespace mindspore {
namespace parallel {
std::string ParameterName(const AnfNodePtr &node_ptr) {
  auto para_ptr = node_ptr->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(para_ptr);
  return para_ptr->name();
}

bool ParameterRequireGrad(const AnfNodePtr &node_ptr) {
  auto para_ptr = node_ptr->cast<ParameterPtr>();
  if (para_ptr == nullptr) {
    return false;
  }
  if (!para_ptr->has_default()) {
    return false;
  }
  auto obj = py::cast(para_ptr->default_param());
  auto param_value = py::cast<ParamValuePtr>(obj.attr("_value"));
  if (param_value == nullptr) {
    return false;
  }
  return param_value->requires_grad();
}
}  // namespace parallel
}  // namespace mindspore
