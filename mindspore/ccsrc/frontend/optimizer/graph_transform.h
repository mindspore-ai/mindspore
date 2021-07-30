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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_GRAPH_TRANSFORM_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_GRAPH_TRANSFORM_H

#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
bool CNodeHasTupleInput(const CNodePtr &cnode);
bool FuncGraphHasTupleInput(const FuncGraphPtr &fg);
std::vector<AnfNodePtr> TransformTupleArgument(const FuncGraphPtr &fg, const AnfNodePtr &node,
                                               const abstract::AbstractTuplePtr &abs);
AnfNodePtr TransformCallGraph(const FuncGraphPtr &trans_fg, const CNodePtr &cnode);
AnfNodePtr TransformPartial(const FuncGraphPtr &trans_fg, const CNodePtr &cnode);
AnfNodePtr TransformSwitchCall(const AnfNodePtr &swtich_node, const CNodePtr &cnode);

class GraphTupleParamTransform {
 public:
  GraphTupleParamTransform() : cache_() {}
  ~GraphTupleParamTransform() { cache_.clear(); }
  FuncGraphPtr operator()(const FuncGraphPtr &fg, const FuncGraphManagerPtr &mng) {
    if (cache_.find(fg) != cache_.end()) {
      return cache_[fg];
    }
    auto new_fg = TransformGraphParam(fg, mng);
    cache_[fg] = new_fg;
    return new_fg;
  }

  AnfNodePtr GenerateTupleParams(const abstract::AbstractTuplePtr &tuple_abs, const FuncGraphPtr &fg,
                                 std::vector<AnfNodePtr> *params) {
    std::vector<AnfNodePtr> inputs;
    inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
    auto &elements = tuple_abs->elements();
    for (auto &item : elements) {
      if (item->isa<abstract::AbstractTuple>()) {
        inputs.push_back(GenerateTupleParams(item->cast<abstract::AbstractTuplePtr>(), fg, params));
      } else {
        auto p = std::make_shared<Parameter>(fg);
        p->set_abstract(item);
        params->push_back(p);
        inputs.push_back(params->back());
      }
    }
    auto node = fg->NewCNode(inputs);
    node->set_abstract(tuple_abs);
    return node;
  }

  FuncGraphPtr TransformGraphParam(const FuncGraphPtr &fg, const FuncGraphManagerPtr &mng) {
    Cloner cloner({fg}, false, false, false, std::make_shared<TraceCopy>(), std::make_shared<TraceCopy>());
    auto new_fg = cloner[fg];
    auto &params = new_fg->parameters();
    std::vector<AnfNodePtr> new_params;
    std::unordered_map<AnfNodePtr, AnfNodePtr> repl;
    for (auto &param : params) {
      auto abs = param->abstract();
      if (abs != nullptr && abs->isa<abstract::AbstractTuple>()) {
        auto tuple_abs = abs->cast<abstract::AbstractTuplePtr>();
        std::vector<AnfNodePtr> tuple_params;
        repl.emplace(param, GenerateTupleParams(tuple_abs, new_fg, &tuple_params));
        std::transform(tuple_params.begin(), tuple_params.end(), std::back_inserter(new_params),
                       [](AnfNodePtr p) { return p; });
      } else {
        new_params.push_back(param);
      }
    }
    auto tmp_mng = mindspore::Manage(new_fg, false);
    auto tr = tmp_mng->Transact();
    for (auto &item : repl) {
      bool ret = tr.Replace(item.first, item.second);
      if (ret == false) {
        MS_LOG(ERROR) << "replace failed" << item.first->DebugString() << " with__" << item.second->DebugString(2);
      }
    }
    tr.SetParameters(new_fg, new_params);
    tr.Commit();
    mng->AddFuncGraph(new_fg);
    return new_fg;
  }

 private:
  std::unordered_map<FuncGraphPtr, FuncGraphPtr> cache_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_GRAPH_TRANSFORM_H
