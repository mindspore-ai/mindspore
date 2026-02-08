/**
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_DO_SIGNATURE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_DO_SIGNATURE_H_

#include <vector>
#include <string>
#include <utility>
#include <map>
#include <set>
#include <memory>

#include "frontend/jit/ps/static_analysis/static_analysis.h"
#include "utils/any.h"
#include "ir/meta_func_graph.h"
#include "utils/ms_utils.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
class DoSignatureMetaFuncGraph : public MetaFuncGraph {
 public:
  explicit DoSignatureMetaFuncGraph(const std::string &name, const ValuePtr &function)
      : MetaFuncGraph("S-" + name), function_(function) {}

  ~DoSignatureMetaFuncGraph() override = default;

  MS_DECLARE_PARENT(DoSignatureMetaFuncGraph, MetaFuncGraph)

  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &args_abs_list) override;
  const ValuePtr function() const { return function_; }

  friend bool operator==(const DoSignatureMetaFuncGraph &lhs, const DoSignatureMetaFuncGraph &rhs) {
    return &lhs == &rhs;
  }

 private:
  ValuePtr function_;
};
using RWSignaturePtr = std::shared_ptr<DoSignatureMetaFuncGraph>;

bool IfRaiseExceptionForCheckParameter(const std::string &func_name, const ValuePtr &function,
                                       const SignatureEnumRW &sig, const TypePtr &type);

std::vector<AnfNodePtr> GetNewInputsBySignatures(const FuncGraphPtr &func_graph, const std::string &func_name,
                                                 const ValuePtr &function, const AbstractBasePtrList &args_abs_list,
                                                 const std::vector<AnfNodePtr> &params_list,
                                                 const AnfNodePtr &old_cnode = nullptr);
AnfNodePtr GenerateCNodeBySignatures(const FuncGraphPtr &func_graph, const std::string &func_name,
                                     const ValuePtr &function, const AbstractBasePtrList &args_abs_list,
                                     const AnfNodePtrList &old_node_inputs);
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_DO_SIGNATURE_H_
