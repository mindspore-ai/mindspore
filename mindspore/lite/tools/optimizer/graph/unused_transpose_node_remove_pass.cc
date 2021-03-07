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
#include "tools/optimizer/graph/unused_transpose_node_remove_pass.h"
#include <vector>
#include <memory>
#include "ops/transpose.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "include/errorcode.h"

namespace mindspore::opt {
static constexpr size_t kTransposeInput = 1;
constexpr size_t kTransposeInputNum = 3;
const std::vector<int> kPermNCHW{0, 3, 1, 2};
const std::vector<int> kPermNHWC{0, 2, 3, 1};
void RemoveUnusedTransposeOpPass::SetFmkType(FmkType type) { this->fmk_type = type; }

std::vector<int> GetTransposePerm(const CNodePtr &node) {
  MS_ASSERT(node != nullptr);
  std::vector<int> perm;
  if (!CheckPrimitiveType(node, prim::kPrimTranspose)) {
    return perm;
  }
  if (node->inputs().size() != kTransposeInputNum) {
    return perm;
  }
  auto perm_node = node->input(2);
  if (!utils::isa<ParameterPtr>(perm_node)) {
    return perm;
  }
  auto perm_param = perm_node->cast<ParameterPtr>();
  if (!perm_param->has_default() || perm_param->default_param() == nullptr) {
    return perm;
  }
  auto perm_value = perm_param->default_param()->cast<ParamValueLitePtr>();
  if (perm_value == nullptr) {
    return perm;
  }
  perm.resize(perm_value->tensor_shape()[0]);
  if (memcpy_s(perm.data(), perm_value->tensor_size(), perm_value->tensor_addr(), perm_value->tensor_size()) != EOK) {
    MS_LOG(ERROR) << "memcpy failed.";
    return {};
  }
  return perm;
}

bool RemoveUnusedTransposeOpPass::Run(const FuncGraphPtr &func_graph) {
  if (this->fmk_type != lite::converter::FmkType_ONNX) {
    MS_LOG(ERROR) << "The framework type of model should be onnx.";
    return RET_ERROR;
  }
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimTranspose)) {
      auto transpose_cnode = node->cast<CNodePtr>();
      if (!CheckPrimitiveType(transpose_cnode->input(kTransposeInput), prim::kPrimConv2DFusion)) {
        continue;
      }
      if (transpose_cnode->inputs().size() != kTransposeInputNum) {
        MS_LOG(ERROR) << "transpose node need have 2 inputs.";
        return false;
      }
      auto perm = GetTransposePerm(transpose_cnode);
      if (perm == kPermNCHW) {
        manager->Replace(transpose_cnode, transpose_cnode->input(1));
      }
    } else if (CheckPrimitiveType(node, prim::kPrimConv2DFusion)) {
      auto conv_node = node->cast<CNodePtr>();
      if (!CheckPrimitiveType(conv_node->input(kTransposeInput), prim::kPrimTranspose)) {
        continue;
      }
      auto transpose_cnode = conv_node->input(kTransposeInput)->cast<CNodePtr>();
      auto perm = GetTransposePerm(transpose_cnode);
      if (perm == kPermNHWC) {
        manager->Replace(transpose_cnode, transpose_cnode->input(1));
      }
    } else {
      continue;
    }
  }
  return true;
}
}  // namespace mindspore::opt
