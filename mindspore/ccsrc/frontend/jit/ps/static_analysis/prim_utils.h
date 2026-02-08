/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_STATIC_ANALYSIS_PRIM_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_STATIC_ANALYSIS_PRIM_UTILS_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <utility>

#include "frontend/jit/ps/static_analysis/evaluator.h"
#include "frontend/jit/ps/static_analysis/prim_to_function.h"
#include "ops/op_def.h"
#include "ops/ops_frontend_func_impl.h"

namespace mindspore {
namespace abstract {
// Get the __init__() arguments of the PrimitivePy object.
AnfNodePtrList GetPrimitiveInitArgs(const PrimitivePyPtr &prim_py, const ops::OpDef *op_def);

bool ValidateArgsType(const AbstractBasePtr &abs_arg, ops::OP_DTYPE type_arg);

bool ValidateArgSpecialType(const std::string &op_name, const AbstractBasePtr &abs, const ops::OpInputArg &op_arg);

AnfNodePtrList GeneratePrimitiveDefaultArgs(const std::string &op_name, const std::vector<AnfNodePtr> &args_list,
                                            const std::vector<ops::OpInputArg> &op_args,
                                            const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func,
                                            const FuncGraphPtr &graph);

void GetKeywordArgsMap(const AbstractBasePtr &input_abs, const std::vector<ops::OpInputArg> &op_args,
                       const AnfNodePtr &input, const FuncGraphPtr &graph, std::map<std::string, AnfNodePtr> *key_map);

AnfNodePtr CheckAndConvertPrimitiveArgs(const PrimitivePtr &prim,
                                        const std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> &args_pair,
                                        const AnalysisEnginePtr &engine, const AnfNodeConfigPtr &out_conf,
                                        bool is_preprocessed);

// Process the primitive's arguments (such as dtype auto-cast, add argument with default-value...),
// then generate the primitive CNode and add it to graph.
// (The returned CNode is without abstract, need to evaluate its abstract manually).
CNodePtr GeneratePrimitiveCNode(const PrimitivePtr &primitive, const ops::OpDef *op_def, const FuncGraphPtr &graph,
                                const AnfNodePtrList &init_args_nodes, const AnfNodePtrList &call_args_nodes,
                                const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func);

std::shared_ptr<Functional> BuildMethodFunctional(const std::string &name);

// Check whether type x is a subtype of model.
bool IsSubtype(const AbstractBasePtr x, const TypePtr model);

bool ValidateArgOptional(const AbstractBasePtr &abs_arg, const ops::OpInputArg &input_arg);

template <typename T>
bool HasAbstractType(const AbstractBasePtr &abs);
}  // namespace abstract
}  // namespace mindspore
#endif  //  MINDSPORE_CCSRC_FRONTEND_JIT_STATIC_ANALYSIS_PRIM_UTILS_H_
