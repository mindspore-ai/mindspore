/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "mindapi/ir/utils.h"
#include "mindapi/src/helper.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "ir/func_graph_cloner.h"
#include "utils/check_convert_utils.h"

namespace mindspore::api::utils {
using ValueImpl = mindspore::Value;
using FuncGraphImpl = mindspore::FuncGraph;

MIND_API FuncGraphPtr CloneGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto fg_impl = ToImpl<FuncGraphImpl>(func_graph);
  Cloner cloner({fg_impl}, false, true, true, std::make_shared<TraceCopy>(), nullptr);
  auto cloned_fg = cloner[fg_impl];
  return ToWrapper<api::FuncGraph>(cloned_fg);
}

int64_t GetPadMode(const api::ValuePtr &value, bool is_upper) {
  int64_t result;
  auto value_impl = ToImpl<ValueImpl>(value);
  CheckAndConvertUtils::GetPadModEnumValue(value_impl, &result, is_upper);
  return result;
}
}  // namespace mindspore::api::utils
