/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/const_fold/fold_along_infershape.h"
#include <memory>
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
STATUS ConstFoldAlongInferShape::PostProcess(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (!CheckCanFold(func_graph, cnode)) {
    return lite::RET_OK;
  }
  if (const_fold_processor_ == nullptr) {
    const_fold_processor_ = std::make_shared<ConstFoldProcessor>(fmk_type_, train_flag_);
  }
  MS_CHECK_TRUE_MSG(const_fold_processor_ != nullptr, lite::RET_NULL_PTR, "const fold processor is nullptr");
  auto status = const_fold_processor_->DoConstantFold(func_graph, cnode);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "do constant fold failed, the node is " << cnode->fullname_with_scope();
  }
  return status;
}

bool ConstFoldAlongInferShape::CheckCanFold(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (IsSpecialType(cnode) || CheckPrimitiveType(cnode, prim::kPrimCustom) || IsMarkedTrainOp(cnode)) {
    return false;
  }
  auto inputs = cnode->inputs();
  auto graph_inputs =
    sub_inputs_map_.find(func_graph) != sub_inputs_map_.end() ? sub_inputs_map_[func_graph] : func_graph->get_inputs();
  auto is_all_const = std::all_of(inputs.begin(), inputs.end(), [&graph_inputs](const AnfNodePtr &node) {
    return (node->isa<ValueNode>() && !IsValueNode<FuncGraph>(node)) ||
           (node->isa<Parameter>() && node->cast<ParameterPtr>()->has_default() &&
            std::find(graph_inputs.begin(), graph_inputs.end(), node) == graph_inputs.end());
  });
  if (is_all_const) {
    return true;
  }
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    return false;
  }
  auto is_inferred = prim->GetAttr(kInferDone) != nullptr && GetValue<bool>(prim->GetAttr(kInferDone));
  if (!is_inferred) {
    return false;
  }
  if (CheckPrimitiveType(cnode, prim::kPrimShape) &&
      lite::ConverterInnerContext::GetInstance()->GetGraphInputTensorShapeMapSize() != 0) {
    return true;
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
