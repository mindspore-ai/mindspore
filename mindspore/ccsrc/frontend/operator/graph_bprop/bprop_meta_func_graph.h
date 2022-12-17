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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_BPROP_META_FUNC_GRAPH_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_BPROP_META_FUNC_GRAPH_H_

#include <string>
#include <memory>
#include <vector>
#include "ir/meta_func_graph.h"

namespace mindspore {
namespace graph_bprop {
class BpropMetaFuncGraph : public MetaFuncGraph {
 public:
  BpropMetaFuncGraph(const std::string &name, const PrimitivePtr &primal) : MetaFuncGraph(name), primal_(primal) {}
  ~BpropMetaFuncGraph() override = default;
  MS_DECLARE_PARENT(BpropMetaFuncGraph, MetaFuncGraph);

  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &input_abs) override {
    MS_LOG(EXCEPTION) << "Should not generate func_graph for the base BpropMetaFuncGraph class.";
  }

 protected:
  PrimitivePtr primal_;
};

using BpropFunction = std::function<FuncGraphPtr(const PrimitivePtr &)>;
using PrimitiveBpropImplMap = mindspore::HashMap<PrimitivePtr, BpropFunction, PrimitiveHasher, PrimitiveEqual>;

PrimitiveBpropImplMap &GetPrimitiveBpropImplMap();

class RegisterPrimitiveBpropHelper {
 public:
  RegisterPrimitiveBpropHelper(const PrimitivePtr &primitive, const BpropFunction &bprop_fn) {
    auto &prim_bprop_impl_map = GetPrimitiveBpropImplMap();
    prim_bprop_impl_map[primitive] = bprop_fn;
  }
  ~RegisterPrimitiveBpropHelper() = default;
};

#define STR(s) #s

#define REGISTER_PRIMITIVE_BPROP_IMPL(name, primitive, bprop_fn, forward_inputs_size)         \
  class BpropMetaFuncGraph##name : public BpropMetaFuncGraph {                                \
   public:                                                                                    \
    explicit BpropMetaFuncGraph##name(const PrimitivePtr &primal)                             \
        : BpropMetaFuncGraph(STR(BpropMetaFuncGraph##name), primal) {}                        \
    ~BpropMetaFuncGraph##name() override = default;                                           \
    MS_DECLARE_PARENT(BpropMetaFuncGraph##name, BpropMetaFuncGraph);                          \
                                                                                              \
    FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &input_abs) override { \
      return bprop_fn(primal_, input_abs);                                                    \
    }                                                                                         \
  };                                                                                          \
  FuncGraphPtr GetBprop##name(const PrimitivePtr &primal) {                                   \
    auto fg = std::make_shared<FuncGraph>();                                                  \
    auto meta_graph = std::make_shared<BpropMetaFuncGraph##name>(primal);                     \
    std::vector<AnfNodePtr> inputs{NewValueNode(meta_graph)};                                 \
    for (size_t i = 0; i < forward_inputs_size; ++i) {                                        \
      (void)inputs.emplace_back(fg->add_parameter());                                         \
    }                                                                                         \
    (void)inputs.emplace_back(fg->add_parameter());                                           \
    (void)inputs.emplace_back(fg->add_parameter());                                           \
    fg->set_output(fg->NewCNode(inputs));                                                     \
    return fg;                                                                                \
  }                                                                                           \
  static auto helper_bprop_##name = RegisterPrimitiveBpropHelper(primitive, GetBprop##name);
}  // namespace graph_bprop
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_BPROP_META_FUNC_GRAPH_H_
