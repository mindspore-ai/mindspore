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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GE_SPECIALIZED_PREPARE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GE_SPECIALIZED_PREPARE_H_

#include <vector>
#include <algorithm>
#include <unordered_map>

#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "ir/pattern_matcher.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"

namespace mindspore {
namespace opt {
namespace irpass {
class GeTensorArrayPrepare {
 public:
  GeTensorArrayPrepare() = default;
  virtual ~GeTensorArrayPrepare() = default;

  bool operator()(const FuncGraphPtr &root, const OptimizerPtr &optimizer) {
    AnfNodePtr ret = root->get_return();
    MS_EXCEPTION_IF_NULL(ret);
    std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);

    bool change = false;
    for (auto &node : all_nodes) {
      if (IsPrimitiveCNode(node, prim::kPrimTensorArray)) {
        TransformTASizeFromAttrToInput(node);
        change = true;
      }
    }
    if (change) {
      InsertFlowOutputToTA(all_nodes);
    }
    return change;
  }

 private:
  // Add a const input with value `size` to TensorArray node
  void TransformTASizeFromAttrToInput(const AnfNodePtr &node);
  void InsertFlowOutputToTA(const std::vector<AnfNodePtr> &all_nodes);
  std::unordered_map<AnfNodePtr, AnfNodePtr> converted_ta_node_;
  std::unordered_map<AnfNodePtr, AbstractBasePtr> ta_node_abstract_cache_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // #ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GE_SPECIALIZED_PREPARE_H_
