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
#include "plugin/device/ascend/optimizer/ir_fusion/layer_norm_beta_gamma_backprop_fusion.h"
#include <memory>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"
namespace mindspore {
namespace opt {
using common::SafeCStr;
namespace {
void GetOutputCastNodes(const FuncGraphPtr &func_graph, const AnfNodePtr &node, std::vector<CNodePtr> *cast_nodes) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(node) == manager->node_users().end()) {
    return;
  }
  for (const auto &node_index : manager->node_users()[node]) {
    AnfNodePtr output = node_index.first;
    auto output_cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(output_cnode);
    if (common::AnfAlgo::GetCNodeName(output_cnode) != prim::kPrimTupleGetItem->name()) {
      MS_LOG(EXCEPTION) << "The output of node " << node->DebugString() << " should be "
                        << prim::kPrimTupleGetItem->name() << trace::DumpSourceLines(node);
    }
    if (manager->node_users().find(output) == manager->node_users().end() ||
        manager->node_users()[output].size() != 1) {
      continue;
    }
    AnfNodePtr transitive_output = manager->node_users()[output].begin()->first;
    MS_EXCEPTION_IF_NULL(transitive_output);
    auto transitive_output_cnode = transitive_output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(transitive_output_cnode);
    if (common::AnfAlgo::GetCNodeName(transitive_output_cnode) == prim::kPrimCast->name()) {
      cast_nodes->push_back(transitive_output_cnode);
    }
  }
}

bool CheckKernelBuildInfo(const CNodePtr &cnode, const kernel::KernelBuildInfoPtr &kernel_info) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(kernel_info);
  for (size_t i = 0; i < kernel_info->GetInputNum(); ++i) {
    if (kernel_info->GetInputDeviceType(i) != kNumberTypeFloat16 ||
        kernel_info->GetInputFormat(i) != AnfAlgo::GetInputFormat(cnode, i)) {
      return false;
    }
  }
  for (size_t i = 0; i < kernel_info->GetOutputNum(); ++i) {
    if (kernel_info->GetOutputDeviceType(i) != kNumberTypeFloat32 ||
        kernel_info->GetOutputFormat(i) != AnfAlgo::GetOutputFormat(cnode, i)) {
      return false;
    }
  }
  return true;
}

bool CheckLayernormBetaGammaBackprop(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                     std::vector<CNodePtr> *cast_nodes) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::HasNodeAttr(kAttrShapeGamma, cnode)) {
    MS_LOG(INFO) << "The node " << cnode->DebugString() << " has no " << kAttrShapeGamma << " attr";
    return false;
  }
  if (common::AnfAlgo::GetInputTensorNum(cnode) != kLayerNormBetaGammaBackpropInputTensorNum) {
    MS_LOG(INFO) << "The node " << cnode->DebugString() << " inputs num is not equal to "
                 << kLayerNormBetaGammaBackpropInputTensorNum;
    return false;
  }
  if (AnfAlgo::GetOutputTensorNum(cnode) != kLayerNormBetaGammaBackpropOutputNum) {
    MS_LOG(INFO) << "The node " << cnode->DebugString() << " outputs num is not equal to "
                 << kLayerNormBetaGammaBackpropOutputNum;
    return false;
  }
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t i = 0; i < input_num; ++i) {
    if (AnfAlgo::GetInputDeviceDataType(cnode, i) != kNumberTypeFloat16) {
      MS_LOG(INFO) << "The data type of node " << cnode->DebugString() << " input " << i << " is not float16";
      return false;
    }
  }
  GetOutputCastNodes(func_graph, cnode, cast_nodes);
  if (cast_nodes->size() != kLayerNormBetaGammaBackpropOutputNum) {
    MS_LOG(INFO) << "The num of cast node in node " << cnode->DebugString() << " outputs is not equal to "
                 << kLayerNormBetaGammaBackpropOutputNum;
    return false;
  }
  for (const auto &cast : *cast_nodes) {
    if (AnfAlgo::GetInputDeviceDataType(cast, 0) != kNumberTypeFloat16 ||
        AnfAlgo::GetOutputDeviceDataType(cast, 0) != kNumberTypeFloat32) {
      MS_LOG(INFO) << "The cast " << cast->DebugString() << " should be fp16->fp32";
      return false;
    }
  }
  return true;
}
}  // namespace

const BaseRef LayerNormBetaGammaBackpropFusion::DefinePattern() const {
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  const auto prim = std::make_shared<Primitive>(kLayerNormBetaGammaBackpropOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr LayerNormBetaGammaBackpropFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                           const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<CNodePtr> cast_nodes;
  if (!CheckLayernormBetaGammaBackprop(func_graph, cnode, &cast_nodes)) {
    return nullptr;
  }
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  MS_EXCEPTION_IF_NULL(kernel_query_);
  kernel_query_->Query(cnode, &kernel_info_list);
  auto alternative_kernel_build_info =
    std::find_if(kernel_info_list.begin(), kernel_info_list.end(),
                 [&cnode](const kernel::KernelBuildInfoPtr &candidate_kernel_build_info) {
                   return CheckKernelBuildInfo(cnode, candidate_kernel_build_info);
                 });
  if (alternative_kernel_build_info == kernel_info_list.end()) {
    MS_LOG(INFO) << "Can not find alternative kernel build info for node " << node->DebugString();
    return nullptr;
  }
  AnfAlgo::SetSelectKernelBuildInfo(*alternative_kernel_build_info, cnode.get());
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // The cast_nodes size has been checked above.
  MS_EXCEPTION_IF_NULL(cast_nodes[0]);
  MS_EXCEPTION_IF_NULL(cast_nodes[1]);
  CheckCNodeInputSize(cast_nodes[0], kCastInputTensorNum);
  CheckCNodeInputSize(cast_nodes[1], kCastInputTensorNum);
  (void)manager->Replace(cast_nodes[0], cast_nodes[0]->input(1));
  (void)manager->Replace(cast_nodes[1], cast_nodes[1]->input(1));
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
