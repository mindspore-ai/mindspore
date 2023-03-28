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
using GetBpropFunction = std::function<FuncGraphPtr(const PrimitivePtr &, const size_t)>;
using BpropFunction = std::function<FuncGraphPtr(const PrimitivePtr &primal, const AbstractBasePtrList &input_abs)>;
using PrimitiveBpropImplMap = mindspore::HashMap<std::string, GetBpropFunction>;

class BpropMetaFuncGraph : public MetaFuncGraph {
 public:
  BpropMetaFuncGraph(const std::string &name, const PrimitivePtr &primal, const BpropFunction &bprop_fn = {})
      : MetaFuncGraph("BpropMetaFuncGraph" + name), primal_(primal), bprop_fn_(bprop_fn) {}
  ~BpropMetaFuncGraph() override = default;
  MS_DECLARE_PARENT(BpropMetaFuncGraph, MetaFuncGraph);

  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &input_abs) override {
    return bprop_fn_(primal_, input_abs);
  }

 protected:
  PrimitivePtr primal_;
  BpropFunction bprop_fn_;
};

PrimitiveBpropImplMap *GetPrimitiveBpropImplMapPtr();
const PrimitiveBpropImplMap &GetPrimitiveBpropImplMap();

class RegisterPrimitiveBpropHelper {
 public:
  RegisterPrimitiveBpropHelper(const std::string &op_name, const GetBpropFunction &get_bprop_fn) {
    auto prim_bprop_impl_map = GetPrimitiveBpropImplMapPtr();
    (*prim_bprop_impl_map)[op_name] = get_bprop_fn;
  }
  ~RegisterPrimitiveBpropHelper() = default;
};

#define STR(s) #s

#define REGISTER_PRIMITIVE_BPROP_IMPL(name, bprop_fn)                                                                  \
  do {                                                                                                                 \
    auto get_bprop_##name = [](const PrimitivePtr &primal, const size_t forward_inputs_size) -> FuncGraphPtr {         \
      auto fg = std::make_shared<FuncGraph>();                                                                         \
      std::vector<AnfNodePtr> inputs{NewValueNode(std::make_shared<BpropMetaFuncGraph>(STR(name), primal, bprop_fn))}; \
      for (size_t i = 0; i < forward_inputs_size; ++i) {                                                               \
        (void)inputs.emplace_back(fg->add_parameter());                                                                \
      }                                                                                                                \
      (void)inputs.emplace_back(fg->add_parameter());                                                                  \
      (void)inputs.emplace_back(fg->add_parameter());                                                                  \
      fg->set_output(fg->NewCNode(inputs));                                                                            \
      return fg;                                                                                                       \
    };                                                                                                                 \
    static auto helper_bprop_##name = graph_bprop::RegisterPrimitiveBpropHelper(STR(name), get_bprop_##name);          \
  } while (0)

void RegArrayOps();
void RegMathOps();
void RegNNOps();
void RegBpropMetaFuncGraph();
}  // namespace graph_bprop
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_BPROP_META_FUNC_GRAPH_H_
