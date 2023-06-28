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

#ifndef MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_BPROP_META_FUNC_GRAPH_H_
#define MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_BPROP_META_FUNC_GRAPH_H_

#include <string>
#include <memory>
#include "ir/meta_func_graph.h"
#include "frontend/expander/bprop/bprop.h"
#include "frontend/expander/utils.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace expander {
namespace bprop {
class BpropMetaFuncGraph : public MetaFuncGraph {
 public:
  explicit BpropMetaFuncGraph(const PrimitivePtr &primal, const BpropHandle *handle)
      : MetaFuncGraph("BpropMetaFuncGraph" + primal->name()), primal_(primal), handle_(handle) {}
  ~BpropMetaFuncGraph() override = default;
  MS_DECLARE_PARENT(BpropMetaFuncGraph, MetaFuncGraph);
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &input_abs) override;

 private:
  PrimitivePtr primal_;
  const BpropHandle *handle_;
};

FuncGraphPtr GetBpropMetaFuncGraph(const PrimitivePtr &primal, const CNodePtr &cnode);
}  // namespace bprop
}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_BPROP_META_FUNC_GRAPH_H_
