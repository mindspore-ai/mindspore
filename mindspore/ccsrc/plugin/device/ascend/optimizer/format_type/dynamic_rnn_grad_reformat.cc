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
#include "plugin/device/ascend/optimizer/format_type/dynamic_rnn_grad_reformat.h"
#include <memory>
#include <string>
#include <utility>

#include "backend/common/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "kernel/kernel_build_info.h"
#include "mindapi/base/type_id.h"
#include "ops/base_operator.h"
#include "ops/core_ops.h"
#include "utils/log_adapter.h"
namespace mindspore {
namespace opt {
const BaseRef DynamicRNNGradReformat::DefinePattern() const {
  VarPtr Xs = std::make_shared<Var>();
  VarPtr Xs2 = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(Xs);
  MS_EXCEPTION_IF_NULL(Xs2);
  const auto split = std::make_shared<Primitive>(prim::kPrimSplitVD->name());
  return VectorRef({split, VectorRef({std::make_shared<Primitive>(prim::kPrimMatMul->name()), Xs, Xs2})});
}

const AnfNodePtr DynamicRNNGradReformat::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto split_v = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(split_v);
  auto matmul = CheckAnfNodeIfCNodeAndInputSize(split_v->input(1), kMatMulInputTensorNum);
  MS_EXCEPTION_IF_NULL(matmul);
  auto input_node_with_idx = common::AnfAlgo::GetPrevNodeOutput(matmul, 0);
  auto input_node = input_node_with_idx.first;
  MS_EXCEPTION_IF_NULL(input_node);
  if (!(input_node->isa<CNode>() &&
        common::AnfAlgo::GetCNodeName(input_node->cast<CNodePtr>()) == kBasicLSTMCellCStateGradV2OpName)) {
    return nullptr;
  }

  // reformat matmul
  auto matmul_kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(matmul);
  MS_EXCEPTION_IF_NULL(matmul_kernel_build_info);
  auto matmul_new_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  matmul_new_builder->SetInputsFormat({kOpFormat_FRAC_NZ, kOpFormat_FRAC_NZ});
  matmul_new_builder->SetOutputsFormat({kOpFormat_FRAC_NZ});
  matmul_new_builder->SetInputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16});
  matmul_new_builder->SetOutputsDeviceType({kNumberTypeFloat});
  matmul_new_builder->SetKernelType(matmul_kernel_build_info->kernel_type());
  matmul_new_builder->SetFusionType(matmul_kernel_build_info->fusion_type());
  matmul_new_builder->SetProcessor(matmul_kernel_build_info->processor());
  AnfAlgo::SetSelectKernelBuildInfo(matmul_new_builder->Build(), matmul.get());
  common::AnfAlgo::SetNodeAttr("insert_backend", MakeValue(true), matmul);

  // reformat split_v
  auto split_kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(split_v);
  MS_EXCEPTION_IF_NULL(split_kernel_build_info);
  auto split_new_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  split_new_builder->SetInputsFormat({kOpFormat_FRAC_NZ});
  split_new_builder->SetOutputsFormat({kOpFormat_FRAC_NZ, kOpFormat_FRAC_NZ});
  split_new_builder->SetInputsDeviceType(split_kernel_build_info->GetAllInputDeviceTypes());
  split_new_builder->SetOutputsDeviceType(split_kernel_build_info->GetAllOutputDeviceTypes());
  split_new_builder->SetKernelType(split_kernel_build_info->kernel_type());
  split_new_builder->SetFusionType(split_kernel_build_info->fusion_type());
  split_new_builder->SetProcessor(split_kernel_build_info->processor());
  AnfAlgo::SetSelectKernelBuildInfo(split_new_builder->Build(), split_v.get());
  common::AnfAlgo::SetNodeAttr("insert_backend", MakeValue(true), split_v);
  return split_v;
}
}  // namespace opt
}  // namespace mindspore
