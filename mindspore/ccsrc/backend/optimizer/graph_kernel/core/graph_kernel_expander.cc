/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/graph_kernel/core/graph_kernel_expander.h"

#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "ir/graph_utils.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "utils/context/graph_kernel_flags.h"
#include "backend/optimizer/graph_kernel/core/graph_builder.h"
#include "backend/optimizer/graph_kernel/core/graph_kernel_callback.h"
#include "backend/optimizer/graph_kernel/core/graph_kernel_utils.h"
#include "backend/optimizer/graph_kernel/expanders/expander_factory.h"

namespace mindspore::graphkernel {
namespace {
std::vector<PrimitivePtr> GetExpandOps() {
  std::vector<OpWithLevel> expand_ops_with_level = {
    {kAllTarget, OpLevel_0, prim::kPrimAddN},
    {kAllTarget, OpLevel_0, prim::kPrimAssignAdd},
    {kAllTarget, OpLevel_0, prim::kPrimErfc},
    {kAllTarget, OpLevel_1, prim::kPrimExpandDims},
    {kAllTarget, OpLevel_0, prim::kPrimGeLU},
    {kAllTarget, OpLevel_0, prim::kPrimGeLUGrad},
    {kAllTarget, OpLevel_0, prim::kPrimSquare},
    {kAllTarget, OpLevel_0, prim::kPrimTile},
    {kAscendDevice, OpLevel_0, prim::kLambApplyOptimizerAssign},
    {kAscendDevice, OpLevel_0, prim::kLambApplyWeightAssign},
    {kAscendDevice, OpLevel_0, prim::kPrimClipByNormNoDivSum},
    {kAscendDevice, OpLevel_0, prim::kPrimSqrtGrad},
    {kAscendDevice, OpLevel_1, prim::kSoftmaxGradExt},
    {kAscendDevice, OpLevel_0, prim::kFusedMulAdd},
    {kGPUDevice, OpLevel_1, prim::kPrimBatchMatMul},
    {kGPUDevice, OpLevel_0, prim::kPrimBiasAdd},
    {kGPUDevice, OpLevel_1, prim::kPrimBiasAddGrad},
    {kGPUDevice, OpLevel_0, prim::kPrimDropout},
    {kGPUDevice, OpLevel_0, prim::kPrimDropoutGrad},
    {kGPUDevice, OpLevel_0, prim::kPrimFusedAdam},
    {kGPUDevice, OpLevel_0, prim::kPrimFusedAdamWeightDecay},
    {kGPUDevice, OpLevel_1, prim::kPrimMaximumGrad},
    {kGPUDevice, OpLevel_1, prim::kPrimMinimumGrad},
    {kGPUDevice, OpLevel_1, prim::kPrimLayerNorm},
    {kGPUDevice, OpLevel_1, prim::kPrimLayerNormGrad},
    {kGPUDevice, OpLevel_0, prim::kPrimLogSoftmax},
    {kGPUDevice, OpLevel_0, prim::kPrimLogSoftmaxGrad},
    {kGPUDevice, OpLevel_1, prim::kPrimMatMul},
    {kGPUDevice, OpLevel_1, prim::kPrimReduceMean},
    {kGPUDevice, OpLevel_0, prim::kPrimRelu},
    {kGPUDevice, OpLevel_0, prim::kPrimReluGrad},
    {kGPUDevice, OpLevel_0, prim::kPrimSigmoid},
    {kGPUDevice, OpLevel_0, prim::kPrimSigmoidGrad},
    {kGPUDevice, OpLevel_0, prim::kPrimSigmoidCrossEntropyWithLogits},
    {kGPUDevice, OpLevel_0, prim::kPrimSigmoidCrossEntropyWithLogitsGrad},
    {kGPUDevice, OpLevel_0, prim::kPrimSlice},
    {kGPUDevice, OpLevel_1, prim::kPrimSoftmax},
    {kGPUDevice, OpLevel_1, prim::kPrimSoftmaxCrossEntropyWithLogits},
    {kGPUDevice, OpLevel_0, prim::kPrimSquaredDifference},
    {kGPUDevice, OpLevel_0, prim::kPrimSqueeze},
    {kGPUDevice, OpLevel_0, prim::kPrimEqualCount},
    {kGPUDevice, OpLevel_0, prim::kPrimSquareSumAll},
    {kGPUDevice, OpLevel_0, prim::kPrimIdentityMath},
    {kGPUDevice, OpLevel_0, prim::kPrimOnesLike},
    {kGPUDevice, OpLevel_0, prim::kPrimStandardNormal},
    {kCPUDevice, OpLevel_0, prim::kPrimOnesLike},
    {kCPUDevice, OpLevel_0, prim::kPrimBiasAdd},
    {kCPUDevice, OpLevel_1, prim::kPrimBiasAddGrad},
    {kCPUDevice, OpLevel_0, prim::kPrimRelu},
    {kCPUDevice, OpLevel_1, prim::kPrimMaximumGrad},
    {kCPUDevice, OpLevel_1, prim::kPrimMinimumGrad},
    {kCPUDevice, OpLevel_1, prim::kPrimAdam},
    {kCPUDevice, OpLevel_1, prim::kPrimTanhGrad},
    {kCPUDevice, OpLevel_1, prim::kPrimSoftplus},
    {kCPUDevice, OpLevel_1, prim::kPrimSoftplusGrad},
  };
  const auto &flags = GraphKernelFlags::GetInstance();
  return GkUtils::GetValidOps(expand_ops_with_level, flags.fusion_ops_level, flags.enable_expand_ops_only,
                              flags.enable_expand_ops, flags.disable_expand_ops);
}
}  // namespace

FuncGraphPtr DefaultExpander::CreateExpandFuncGraph(const CNodePtr &node) {
  auto expander_ptr = expanders::OpExpanderFactory::Instance().GetExpander(AnfUtils::GetCNodeName(node));
  if (expander_ptr == nullptr) {
    MS_LOG(INFO) << "expander not found " << node->fullname_with_scope();
    return nullptr;
  }
  expanders::BaseInfoList inputs(node->size() - 1);
  expanders::BaseInfoList outputs(AnfUtils::GetOutputTensorNum(node));
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  for (size_t i = 0; i < inputs.size(); i++) {
    inputs[i].shape = cb->GetInputShape(node, i);
    inputs[i].type = cb->GetInputType(node, i);
    inputs[i].format = cb->GetInputFormat(node, i);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs[i].shape = cb->GetOutputShape(node, i);
    outputs[i].type = cb->GetOutputType(node, i);
    outputs[i].format = cb->GetOutputFormat(node, i);
  }
  auto &attrs = GetCNodePrimitive(node)->attrs();
  auto litegraph = expander_ptr->Run(inputs, outputs, attrs, cb->GetProcessor(node));
  if (litegraph == nullptr) {
    MS_LOG(INFO) << "undo expanding " << node->fullname_with_scope();
    return nullptr;
  }
  return GkUtils::LiteGraph2AnfGraph(litegraph);
}

AnfNodePtr DefaultExpander::CreateExpandGraphKernel(const FuncGraphPtr &new_func_graph, const CNodePtr &old_node) {
  auto func_graph = old_node->func_graph();
  std::vector<AnfNodePtr> inputs(old_node->inputs().begin() + 1, old_node->inputs().end());
  auto graph_kernel_node = CreateNewFuseCNode(func_graph, new_func_graph, inputs);
  MS_LOG(DEBUG) << "Expand node: " << old_node->fullname_with_scope()
                << " with: " << graph_kernel_node->fullname_with_scope();
  return graph_kernel_node;
}

AnfNodePtr DefaultExpander::Run(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto new_func_graph = CreateExpandFuncGraph(cnode);
  if (new_func_graph == nullptr) {
    return nullptr;
  }
  new_func_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(AnfUtils::GetCNodeName(cnode)));
  auto graph_kernel_node = CreateExpandGraphKernel(new_func_graph, cnode);
  if (AnfUtils::GetOutputTensorNum(node) != AnfUtils::GetOutputTensorNum(graph_kernel_node)) {
    MS_LOG(ERROR) << "The output num of composite node (" << AnfUtils::GetOutputTensorNum(graph_kernel_node)
                  << ") does not match the original basic node (" << AnfUtils::GetOutputTensorNum(node) << ")."
                  << node->fullname_with_scope();
    return nullptr;
  }
  return graph_kernel_node;
}

ExpanderPtr GraphKernelExpander::GetExpander(const AnfNodePtr &) { return std::make_shared<DefaultExpander>(); }

bool GraphKernelExpander::DoExpand(const FuncGraphPtr &func_graph) {
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  std::reverse(todos.begin(), todos.end());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  for (const auto &n : todos) {
    auto node = n->cast<CNodePtr>();
    if (node == nullptr || AnfUtils::IsGraphKernel(node) || GkUtils::IsKeepBasicNode(node) ||
        !AnfUtils::IsRealKernel(node) || !CanExpand(node)) {
      continue;
    }

    MS_LOG(DEBUG) << "Expanding node: " << node->fullname_with_scope();
    auto new_node = GetExpander(node)->Run(node);
    if (new_node == nullptr) {
      MS_LOG(DEBUG) << "Skipped node: " << node->fullname_with_scope();
      continue;
    }
    (void)mng->Replace(node, new_node);
    changed = true;
  }
  return changed;
}

bool GraphKernelExpander::Run(const FuncGraphPtr &func_graph) {
  expand_ops_ = GetExpandOps();
  return DoExpand(func_graph);
}
}  // namespace mindspore::graphkernel
