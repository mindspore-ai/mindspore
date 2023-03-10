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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_HELPER_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_HELPER_H_

#include <utility>
#include <memory>
#include <vector>
#include "include/backend/optimizer/helper.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
class Helper {
 public:
  static std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedList(const FuncGraphPtr &graph,
                                                                                      const AnfNodePtr &node);
  static std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedListByOutputIdx(
    const FuncGraphPtr &graph, const AnfNodePtr &node, size_t output_index);
  static AnfNodePtr SexpToNode(const BaseRef &sexp, const BaseRef &graph, PrimitiveVarMap *primitive_vars,
                               bool multigraph);

 private:
  static ValueNodePtr CreateValueNodeWithSexp(const BaseRef &sexp);
  static CNodePtr CreateCNodeWithGraph(const std::vector<AnfNodePtr> &input_nodes, const BaseRef &graph);
  static VarNodePtr CreateVarNodeWithSexp(const BaseRef &sexp, const BaseRef &graph);
  static AnfNodePtr HandleSexpVector(const BaseRef &sexp, const BaseRef &graph, PrimitiveVarMap *primitive_vars,
                                     bool multigraph);
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_HELPER_H_
