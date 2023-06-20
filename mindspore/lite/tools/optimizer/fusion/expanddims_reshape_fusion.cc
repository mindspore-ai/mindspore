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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/expanddims_reshape_fusion.h"
#include "mindspore/core/ops/array_ops.h"
#include "tools/lite_exporter/fetch_content.h"
#include "ops/op_utils.h"
#include "ops/reshape.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "include/registry/converter_context.h"

namespace mindspore::opt {
const BaseRef ExpandDimsReshapeFusion::DefinePattern() const {
  auto is_reshape = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto reshape_shape = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_shape != nullptr, {});
  auto is_expanddims = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimExpandDims>);
  MS_CHECK_TRUE_RET(is_expanddims != nullptr, {});
  return VectorRef({is_reshape, is_expanddims, reshape_shape});
}

bool ExpandDimsReshapeFusion::CheckCanFuse(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  auto reshape_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr, false);

  MS_CHECK_TRUE_RET(reshape_cnode->input(SECOND_INPUT) != nullptr, false);
  auto expanddims_cnode = reshape_cnode->input(SECOND_INPUT)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(expanddims_cnode != nullptr, false);
  if (IsMultiOutputTensors(func_graph, expanddims_cnode)) {
    return false;
  }
  auto expanddims_primc = GetCNodePrimitive(expanddims_cnode);
  MS_CHECK_TRUE_RET(expanddims_primc != nullptr, false);
  if (IsQuantParameterNode(expanddims_primc)) {
    MS_LOG(INFO) << expanddims_primc->name() << " is quant node";
    return false;
  }
  return true;
}

const AnfNodePtr ExpandDimsReshapeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }

  if (!CheckCanFuse(func_graph, node)) {
    return nullptr;
  }

  auto reshape_cnode = node->cast<CNodePtr>();
  auto expanddims_cnode = reshape_cnode->input(SECOND_INPUT)->cast<CNodePtr>();
  auto manage = Manage(func_graph);
  MS_CHECK_TRUE_RET(manage != nullptr, nullptr);
  manage->SetEdge(reshape_cnode, C1NUM, expanddims_cnode->input(SECOND_INPUT));
  return reshape_cnode;
}
}  // namespace mindspore::opt
