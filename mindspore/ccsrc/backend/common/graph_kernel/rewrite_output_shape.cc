/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/rewrite_output_shape.h"

#include <vector>
#include <memory>
#include "ir/scalar.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::graphkernel {
bool SaveOutputShape::Run(const FuncGraphPtr &func_graph) {
  if (!IsPrimitiveCNode(func_graph->output(), prim::kPrimMakeTuple)) {
    // the MakeTuple with kMetaTypeNone
    return false;
  }

  auto last_maketuple = func_graph->output()->cast<CNodePtr>();
  const auto kMultiInputsNum = 3;
  if (last_maketuple->inputs().size() >= kMultiInputsNum) {
    // MakeTuple of multi inputs
    return false;
  }
  auto output = last_maketuple->input(1)->cast<CNodePtr>();
  if (output == nullptr) {
    return false;
  }
  if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    // the MakeTuple of multi output
    return false;
  }
  // single output, add a MakeTuple node with attr "graph_kernel",
  // this node will be deleted in RewriteOutputShape
  AnfNodePtrList mt_inputs = {NewValueNode(prim::kPrimMakeTuple), output};
  auto mt_node = func_graph->NewCNode(mt_inputs);
  AbstractBasePtrList abstracts = {output->abstract()};
  mt_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstracts));
  SetNodeAttrSafely("graph_kernel", MakeValue(true), mt_node);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  (void)mng->Replace(output, mt_node);
  return true;
}

void RewriteOutputShape::Process(const AnfNodePtr &node, size_t index, const AbstractBasePtr &abstract) {
  if (!node->isa<CNode>()) {
    return;
  }
  // the previous nodes should not be MakeTuple again.
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    return;
  }

  if (node->abstract()->isa<abstract::AbstractTuple>()) {
    auto node_abstracts = node->abstract()->cast<abstract::AbstractTuplePtr>()->elements();
    if (index >= node_abstracts.size()) {
      MS_LOG(EXCEPTION) << "Index " << index << " is out of the range of node_abstracts [0, " << node_abstracts.size()
                        << ") in node " << node->fullname_with_scope();
    }
    node_abstracts[index] = abstract;
    node->set_abstract(std::make_shared<abstract::AbstractTuple>(node_abstracts));
  } else {
    node->set_abstract(abstract);
  }

  // do not process from real kernel
  if (AnfUtils::IsRealKernel(node)) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() <= 1) {
    return;
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
    auto idx = GetValue<int64_t>(GetValueNode(cnode->input(kInputNodeOutputIndexInTupleGetItem)));
    Process(cnode->input(kRealInputNodeIndexInTupleGetItem), LongToSize(idx), abstract);
  } else {
    Process(cnode->input(1), 0, abstract);
  }
}

bool RewriteOutputShape::Run(const FuncGraphPtr &func_graph) {
  if (!IsPrimitiveCNode(func_graph->output(), prim::kPrimMakeTuple)) {
    // the MakeTuple with kMetaTypeNone
    return false;
  }
  auto output = func_graph->output()->cast<CNodePtr>()->input(1)->cast<CNodePtr>();
  if (output == nullptr) {
    return false;
  }
  if (!IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    return false;
  }
  auto abs_tuple = dyn_cast<abstract::AbstractTuple>(output->abstract());
  MS_EXCEPTION_IF_NULL(abs_tuple);
  if (abs_tuple->elements().size() + 1 != output->size()) {
    MS_LOG(EXCEPTION) << "Size of abstract elements does not match the MakeTuple's input size: "
                      << abs_tuple->elements().size() << " vs " << output->size();
  }

  for (size_t i = 1; i < output->size(); i++) {
    Process(output->input(i), 0, abs_tuple->elements()[i - 1]);
  }
  // remove the MakeTuple node if it was added by SaveOutputShape
  auto prim = common::AnfAlgo::GetCNodePrimitive(output);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->HasAttr("graph_kernel")) {
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    (void)mng->Replace(output, output->input(1));
  }
  return true;
}
}  // namespace mindspore::graphkernel
