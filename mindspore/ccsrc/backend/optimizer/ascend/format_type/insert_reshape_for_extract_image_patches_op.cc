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

#include "backend/optimizer/ascend/format_type/insert_reshape_for_extract_image_patches_op.h"
#include <memory>
#include "backend/optimizer/ascend/ascend_helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "base/core_ops.h"

namespace mindspore {
namespace opt {
const BaseRef InsertReshapeForExtractImagePatchesOp::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimExtractImagePatches, Xs});
}

const AnfNodePtr InsertReshapeForExtractImagePatchesOp::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                                const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  auto extract = CheckAnfNodeIfCNodeAndInputSize(node, 2);
  MS_EXCEPTION_IF_NULL(extract);
  auto in_node = extract->input(1);
  MS_EXCEPTION_IF_NULL(in_node);
  auto extract_kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(extract);
  auto in_node_kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(in_node);
  MS_EXCEPTION_IF_NULL(extract_kernel_build_info);
  MS_EXCEPTION_IF_NULL(in_node_kernel_build_info);
  std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())),
                                            in_node};
  auto reshape_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  reshape_builder->SetInputsFormat({kOpFormat_NC1HWC0});
  reshape_builder->SetOutputsFormat({kOpFormat_NC1HWC0});
  reshape_builder->SetInputsDeviceType({AnfAlgo::GetOutputDeviceDataType(in_node, 0)});
  reshape_builder->SetOutputsDeviceType({AnfAlgo::GetOutputDeviceDataType(in_node, 0)});
  reshape_builder->SetKernelType(in_node_kernel_build_info->kernel_type());
  reshape_builder->SetFusionType(in_node_kernel_build_info->fusion_type());
  reshape_builder->SetProcessor(in_node_kernel_build_info->processor());

  auto reshape = func_graph->NewCNode(reshape_inputs);
  reshape->set_scope(in_node->scope());
  auto shape_tmp = AnfAlgo::GetOutputInferShape(in_node, 0);
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputDeviceDataType(in_node, 0)},
                                      {{shape_tmp[0], shape_tmp[2], shape_tmp[3], shape_tmp[1]}}, reshape.get());
  AnfAlgo::SetSelectKernelBuildInfo(reshape_builder->Build(), reshape.get());
  AnfAlgo::SetNodeAttr("nop_op", MakeValue(true), reshape);
  AnfAlgo::SetNodeInput(extract, reshape, 0);
  return extract;
}
}  // namespace opt
}  // namespace mindspore
