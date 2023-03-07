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
#include "plugin/device/ascend/optimizer/ir_fusion/deformable_offsets_grad_fusion.h"
#include <memory>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kAxisH = 2;
constexpr size_t kAxisW = 3;
constexpr size_t kAxisC = 1;
constexpr size_t kDeformableOffsetsGradInputNum = 4;
}  // namespace

const BaseRef DeformableOffsetsGradFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimDeformableOffsetsGrad, Xs});
}

const AnfNodePtr DeformableOffsetsGradFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                      const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  auto deformable_offsets_grad_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(deformable_offsets_grad_cnode);
  size_t origin_input_size = deformable_offsets_grad_cnode->inputs().size();
  if (origin_input_size <= kDeformableOffsetsGradInputNum) {
    MS_LOG(INFO) << "The node " << deformable_offsets_grad_cnode->DebugString() << " is not equal to "
                 << kDeformableOffsetsGradInputNum << " inputs";
  }
  auto pads = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(deformable_offsets_grad_cnode, kAttrPads);
  auto stride = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(deformable_offsets_grad_cnode, kAttrStrides);
  auto dialation = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(deformable_offsets_grad_cnode, kAttrDilations);
  auto kernel_size = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(deformable_offsets_grad_cnode, kAttrKsize);
  auto offset_shape = common::AnfAlgo::GetOutputInferShape(deformable_offsets_grad_cnode->inputs()[kIndex3], 0);
  std::vector<AnfNodePtr> new_inputs{
    NewValueNode(std::make_shared<Primitive>(prim::kPrimDeformableOffsetsGrad->name()))};
  auto assist_const =
    CreateHelperNode(func_graph, offset_shape, kernel_size, stride, pads, dialation, kAxisH, kAxisW, kAxisC);
  (void)new_inputs.insert(new_inputs.end(), deformable_offsets_grad_cnode->inputs().begin() + 1,
                          deformable_offsets_grad_cnode->inputs().end());
  new_inputs.push_back(assist_const);
  auto new_cnode = NewCNode(new_inputs, func_graph);
  new_cnode->set_abstract(deformable_offsets_grad_cnode->abstract());
  new_cnode->set_scope(deformable_offsets_grad_cnode->scope());
  common::AnfAlgo::CopyNodeAttrs(deformable_offsets_grad_cnode, new_cnode);
  common::AnfAlgo::SetNodeAttr(kAttrDataFormat, MakeValue("NHWC"), new_cnode);
  if (!CheckAICoreSupportedAny(new_cnode)) {
    MS_LOG(INFO) << "DeformableOffsetsGrad failed, return to aicpu.";
    return nullptr;
  }
  if (kernel_graph != nullptr) {
    kernel_graph->AddValueNodeToGraph(assist_const);
    MS_LOG(INFO) << "Add assist tensor for DeformableOffsets op success.";
  }
  return new_cnode;
}
}  // namespace opt
}  // namespace mindspore
