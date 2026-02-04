/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "utils/hash_map.h"
#include "primitive/sequence_ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
namespace opt {
bool FuncGraphHasSequenceInput(const FuncGraphPtr &fg);
bool FuncGraphHasConstantSequenceInput(const FuncGraphPtr &fg);
bool IsSequenceExpandable(const AbstractBasePtr &abs);
std::vector<AnfNodePtr> TransformSequenceArgument(const FuncGraphPtr &fg, const AnfNodePtr &node,
                                                  const abstract::AbstractSequencePtr &abs);
bool ContainSparseTensor(const abstract::AbstractBasePtr &abs);
bool ParamContainSparseTensor(const AnfNodePtr &param);

class GraphSequenceParamTransform {
 public:
  GraphSequenceParamTransform() : cache_() {}
  ~GraphSequenceParamTransform() { cache_.clear(); }
  FuncGraphPtr operator()(const FuncGraphPtr &fg, const FuncGraphManagerPtr &mng) {
    if (cache_.find(fg) != cache_.end()) {
      return cache_[fg];
    }
    auto new_fg = TransformGraphParam(fg, mng);
    cache_[fg] = new_fg;
    return new_fg;
  }

  AnfNodePtr GenerateSequenceParams(const abstract::AbstractSequencePtr &seq_abs, const FuncGraphPtr &fg,
                                    std::vector<AnfNodePtr> *params) {
    std::vector<AnfNodePtr> inputs;
    MS_EXCEPTION_IF_NULL(seq_abs);
    auto prim_sequence = seq_abs->isa<abstract::AbstractTuple>() ? prim::kPrimMakeTuple : prim::kPrimMakeList;
    inputs.push_back(NewValueNode(prim_sequence));
    auto &elements = seq_abs->elements();
    for (auto &item : elements) {
      if (item->isa<abstract::AbstractSequence>() && !item->cast<abstract::AbstractSequencePtr>()->dynamic_len()) {
        inputs.push_back(GenerateSequenceParams(item->cast<abstract::AbstractSequencePtr>(), fg, params));
      } else {
        auto p = std::make_shared<Parameter>(fg);
        p->set_abstract(item);
        params->push_back(p);
        inputs.push_back(params->back());
      }
    }
    auto node = fg->NewCNode(inputs);
    node->set_abstract(seq_abs);
    return node;
  }

  FuncGraphPtr TransformGraphParam(const FuncGraphPtr &fg, const FuncGraphManagerPtr &mng) {
    Cloner cloner({fg}, false, false, false, MakeTraceInfo<TraceCopy>(), MakeTraceInfo<TraceCopy>());
    auto new_fg = cloner[fg];
    auto &params = new_fg->parameters();
    std::vector<AnfNodePtr> new_params;
    mindspore::HashMap<AnfNodePtr, AnfNodePtr> repl;
    for (auto &param : params) {
      auto abs = param->abstract();
      if (IsSequenceExpandable(abs)) {
        auto sequence_abs = abs->cast<abstract::AbstractSequencePtr>();
        std::vector<AnfNodePtr> sequence_params;
        (void)repl.emplace(param, GenerateSequenceParams(sequence_abs, new_fg, &sequence_params));
        std::transform(sequence_params.begin(), sequence_params.end(), std::back_inserter(new_params),
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
        MS_LOG(ERROR) << "replace failed" << item.first->DebugString() << " with__"
                      << item.second->DebugString(AnfNode::DebugStringLevel::kLevel2);
      }
    }
    tr.SetParameters(new_fg, new_params);
    tr.Commit();
    mng->AddFuncGraph(new_fg);
    return new_fg;
  }

 private:
  mindspore::HashMap<FuncGraphPtr, FuncGraphPtr> cache_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_GRAPH_TRANSFORM_H
