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
#include "common/graph_kernel/bprop/bprop.h"

#include <algorithm>
#include <memory>
#include "common/graph_kernel/bprop/expander/infer.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace expander {
namespace bprop {
class BpropExpander {
 public:
  BpropExpander(CNodePtrList *outputs, DoutUser *dout_user) : outputs_(outputs), dout_user_(dout_user) {}
  ~BpropExpander() = default;

  NodePtrList ExtractInputs(const CNodePtr &cnode, const BpropIRBuilderPtr &ir_builder) {
    NodePtrList nodes;
    nodes.reserve(cnode->size());
    (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), std::back_inserter(nodes),
                         [ir_builder](const AnfNodePtr &no) { return std::make_shared<Node>(no, ir_builder.get()); });
    return nodes;
  }

  bool Run(const CNodePtr &cnode) {
    auto infer = std::make_shared<CppInfer>();
    auto name = AnfUtils::GetCNodeName(cnode);
    auto ir_builder = std::make_shared<BpropIRBuilder>(name, cnode->func_graph(), infer);
    auto inputs = ExtractInputs(cnode, ir_builder);
    auto &attrs = GetCNodePrimitive(cnode)->attrs();
    return ir_builder->Run(inputs, attrs, outputs_, dout_user_);
  }

 private:
  CNodePtrList *outputs_;
  expander::bprop::DoutUser *dout_user_;
};
}  // namespace bprop
}  // namespace expander

void BuildBprop(const CNodePtr &cnode, CNodePtrList *outputs, expander::bprop::DoutUser *dout_user) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(dout_user);
  expander::bprop::BpropExpander e(outputs, dout_user);
  (void)e.Run(cnode);
}
}  // namespace mindspore
