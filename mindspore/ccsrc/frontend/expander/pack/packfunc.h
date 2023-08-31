/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_PACK_FUNC_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_PACK_FUNC_H_

#include <vector>
#include <set>

#include "ir/anf.h"
#include "abstract/abstract_value.h"
#include "pybind_api/ir/primitive_py.h"

namespace mindspore {
namespace expander {
FuncGraphPtr ExpandPackFuncGraph(const PrimitivePtr &prim, const abstract::AbstractBasePtrList &abs_list);
FuncGraphPtr ExpandPackFuncPynative(const PrimitivePtr &prim, const abstract::AbstractBasePtrList &abs_list,
                                    bool pynative_grad = false);
void ClearAllPackCache();
void ClearCompileAllCache();
bool IsPackGraph(const FuncGraphPtr &fg);
void GetPackGraphParams(const FuncGraphPtr &fg, std::vector<AnfNodePtr> *parameters);
void GetSubPackGraphParams(const FuncGraphPtr &fg, const FuncGraphPtr &g, std::vector<AnfNodePtr> *parameters,
                           std::set<const AnfNode *> *memo);
FuncGraphPtr UpdateReusingGraphForPack(const FuncGraphPtr &reusing_graph, const std::vector<AnfNodePtr> &parameters);
}  // namespace expander
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_PACK_FUNC_H_
