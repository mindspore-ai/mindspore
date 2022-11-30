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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_UTILS_H_

#include <string>
#include <vector>
#include "ir/primitive.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace graph_bprop {
ValuePtr GetAndCheckAttr(const PrimitivePtr &prim, const std::string &key);

template <typename T>
inline T GetAttr(const PrimitivePtr &prim, const std::string &key) {
  auto value = GetAndCheckAttr(prim, key);
  return GetValue<T>(value);
}

PrimitivePtr NewPrimitive(const PrimitivePtr &prim, const mindspore::HashMap<std::string, ValuePtr> &attrs = {});

PrimitivePtr NewPrimitive(const std::string &prim_name, const mindspore::HashMap<std::string, ValuePtr> &attrs = {});

FuncGraphPtr NewGraph(const AbstractBasePtrList &abs_list);

void CheckArgSize(const std::vector<AnfNodePtr> &parameters, const AbstractBasePtrList &input_abs,
                  const PrimitivePtr &prim, size_t expected_size);

TypeId GetTensorDType(const AbstractBasePtr &abs);

AnfNodePtr GetClassType(const std::string &package, const std::string &class_name);

bool ConvertToTensor(const AnfNodePtr &node);
}  // namespace graph_bprop
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_UTILS_H_
