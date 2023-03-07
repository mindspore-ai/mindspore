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

#include "plugin/device/ascend/optimizer/ir_fusion/transpose_transdata_fusion.h"
#include <memory>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace opt {
const BaseRef TransposeTransDataFusion::DefinePattern() const {
  const auto prim_transdata = std::make_shared<Primitive>(prim::kPrimTransData->name());
  VectorRef transpose({prim::kPrimTransposeD, input_varptr_});

  return VectorRef({prim_transdata, transpose});
}

const AnfNodePtr TransposeTransDataFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  auto transdata_cnode = CheckAnfNodeIfCNodeAndInputSize(node, kBackendTransposeInputTensorNum);
  MS_EXCEPTION_IF_NULL(transdata_cnode);
  auto transpose_cnode = CheckAnfNodeIfCNodeAndInputSize(transdata_cnode->input(1), kTransOpInputTensorNum);
  MS_EXCEPTION_IF_NULL(transpose_cnode);
  auto transpose_kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(transpose_cnode);
  auto transdata_kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(transdata_cnode);
  MS_EXCEPTION_IF_NULL(transpose_kernel_build_info);
  MS_EXCEPTION_IF_NULL(transdata_kernel_build_info);

  auto new_transdata_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(new_transdata_builder);
  auto transpose_input_formats = transpose_kernel_build_info->GetAllInputFormats();
  new_transdata_builder->SetInputsFormat(transpose_input_formats);
  new_transdata_builder->SetOutputsFormat(transdata_kernel_build_info->GetAllOutputFormats());
  new_transdata_builder->SetInputsDeviceType(transdata_kernel_build_info->GetAllInputDeviceTypes());
  new_transdata_builder->SetOutputsDeviceType(transdata_kernel_build_info->GetAllOutputDeviceTypes());
  new_transdata_builder->SetKernelType(transdata_kernel_build_info->kernel_type());
  new_transdata_builder->SetFusionType(transdata_kernel_build_info->fusion_type());
  new_transdata_builder->SetProcessor(transdata_kernel_build_info->processor());

  auto new_fusion_transdata = std::make_shared<Primitive>(kTransDataOpName);
  if (CheckAICoreSupportedSpec(transdata_cnode, new_transdata_builder->Build())) {
    std::vector<AnfNodePtr> inputs = {NewValueNode(new_fusion_transdata),
                                      utils::cast<AnfNodePtr>((*equiv)[input_varptr_])};
    auto new_node = NewCNode(inputs, func_graph);
    MS_EXCEPTION_IF_NULL(new_node);
    new_node->set_abstract(node->abstract());
    common::AnfAlgo::CopyNodeAttrs(transdata_cnode, new_node);
    common::AnfAlgo::SetNodeAttr(kAttrSrcFormat, MakeValue(transpose_input_formats[0]), new_node);
    AnfAlgo::SetSelectKernelBuildInfo(new_transdata_builder->Build(), new_node.get());
    MS_LOG(INFO) << "transpose transdata fusion node:" << node->fullname_with_scope() << " success";
    return new_node;
  } else {
    MS_LOG(INFO) << "transpose transdata fusion node:" << node->fullname_with_scope() << " failed";
    return node;
  }
}
}  // namespace opt
}  // namespace mindspore
