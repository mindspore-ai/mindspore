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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_BPROP_EXPANDER_META_FUNC_GRAPH_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_BPROP_EXPANDER_META_FUNC_GRAPH_H_

#include <string>
#include <memory>
#include <vector>
#include "ir/meta_func_graph.h"
#include "frontend/operator/graph_bprop/bprop_meta_func_graph.h"
#include "pipeline/pynative/grad/bprop_expander/bprop.h"
#include "frontend/optimizer/expander.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace graph_bprop {
constexpr int64_t kTwo = 2;
constexpr int64_t kOne = 1;
class BpropExpanderMetaFuncGraph : public BpropMetaFuncGraph {
 public:
  explicit BpropExpanderMetaFuncGraph(const PrimitivePtr &primal) : BpropMetaFuncGraph(primal->name(), primal) {}
  ~BpropExpanderMetaFuncGraph() override = default;
  MS_DECLARE_PARENT(BpropExpanderMetaFuncGraph, BpropMetaFuncGraph);
  FuncGraphPtr BpropExpanderFunc(const AbstractBasePtrList &args_spec_list);
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &input_abs) override;
};

FuncGraphPtr GetExpandBprop(const PrimitivePtr &primal, const size_t &forward_inputs_size);

#define STR(s) #s
#define REGISTER_EXPANDER_BPROP_IMPL(name) \
  static auto helper_expand_bprop_##name = graph_bprop::RegisterPrimitiveBpropHelper(STR(name), GetExpandBprop);

void RegBpropExpanderOps();
}  // namespace graph_bprop
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_BPROP_EXPANDER_META_FUNC_GRAPH_H_
