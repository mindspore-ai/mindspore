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

#ifndef MINDSPORE_CCSRC_OPERATOR_COMPOSITE_MAP_H_
#define MINDSPORE_CCSRC_OPERATOR_COMPOSITE_MAP_H_

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "ir/dtype.h"
#include "ir/meta_func_graph.h"
#include "operator/composite/multitype_funcgraph.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using ArgsPairList = std::vector<std::pair<AnfNodePtr, TypePtr>>;

class Map : public MetaFuncGraph {
 public:
  explicit Map(const std::shared_ptr<MultitypeFuncGraph> &fn_leaf = nullptr)
      : MetaFuncGraph("map"),
        fn_leaf_(fn_leaf),
        broadcast_(false),
        nonleaf_({kObjectTypeList, kObjectTypeTuple, kObjectTypeClass}) {
    Init();
  }
  Map(const Map &h) : MetaFuncGraph("map"), fn_leaf_(h.fn_leaf_), broadcast_(h.broadcast_), nonleaf_(h.nonleaf_) {
    Init();
  }
  Map &operator=(const Map &h) {
    if (this != &h) {
      fn_leaf_ = h.fn_leaf_;
      broadcast_ = h.broadcast_;
      nonleaf_ = h.nonleaf_;
      if (fn_leaf_) {
        name_ = "map[" + fn_leaf_->name() + "]";
      }
    }
    return *this;
  }
  ~Map() override = default;
  MS_DECLARE_PARENT(Map, MetaFuncGraph)
  abstract::AbstractBasePtrList NormalizeArgs(const abstract::AbstractBasePtrList &args_spec_list) const override;
  FuncGraphPtr GenerateFromTypes(const TypePtrList &args_spec_list) override;
  MetaFuncGraphPtr GetFnLeaf() { return fn_leaf_; }

 private:
  FuncGraphPtr GenerateLeafFunc(const size_t &args_size);
  AnfNodePtr FullMakeLeaf(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const AnfNodePtrList &args);
  AnfNodePtr FullMakeList(const std::shared_ptr<List> &type, const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                          const ArgsPairList &arg_pairs);
  AnfNodePtr FullMakeTuple(const std::shared_ptr<Tuple> &type, const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                           const ArgsPairList &arg_pairs);
  AnfNodePtr FullMakeClass(const std::shared_ptr<Class> &type, const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                           const ArgsPairList &arg_pairs);
  AnfNodePtr Make(const FuncGraphPtr &graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_pairs);
  void Init() {
    if (fn_leaf_ != nullptr) {
      name_ = "map[" + fn_leaf_->name() + "]";
    }
    signatures_ =
      // def map(func:read, *args:ref):
      std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"args", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindVarPositional}});
  }

  MultitypeFuncGraphPtr fn_leaf_;
  bool broadcast_;
  std::set<TypeId> nonleaf_;
};
using MapPtr = std::shared_ptr<Map>;
class MapPy : public Map {
 public:
  explicit MapPy(const std::shared_ptr<MultitypeFuncGraph> &fn_leaf = nullptr) : Map(fn_leaf) {}
  ~MapPy() override = default;
  MS_DECLARE_PARENT(MapPy, Map)
};
using MapPyPtr = std::shared_ptr<MapPy>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_OPERATOR_COMPOSITE_MAP_H_
