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
#include "tools/optimizer/graph/unused_transpose_node_remove_pass.h"
#include <vector>
#include <memory>
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/lite/include/errorcode.h"
#include "src/ops/primitive_c.h"

namespace mindspore::opt {
static constexpr size_t kTransposeInput = 1;
const std::vector<int> kPermNCHW{0, 3, 1, 2};
const std::vector<int> kPermNHWC{0, 2, 3, 1};
void RemoveUnusedTransposeOpPass::SetFmkType(FmkType type) { this->fmk_type = type; }

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
    auto type = opt::GetCNodeType(node);
    if (type == schema::PrimitiveType_Transpose) {
      auto transpose_cnode = node->cast<CNodePtr>();
      auto typeInput = opt::GetCNodeType(transpose_cnode->input(kTransposeInput));
      if (typeInput != schema::PrimitiveType_Conv2D) {
        continue;
      }
      auto primPtr = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(transpose_cnode->input(0));
      if (primPtr == nullptr) {
        MS_LOG(ERROR) << "Transpose node of onnx need to removed which has not primitiveC";
        return RET_ERROR;
      }
      auto primT = primPtr->GetPrimitiveT();
      if (primT == nullptr) {
        MS_LOG(ERROR) << "Transpose node of onnx need to removed which has not primitiveC";
        return RET_ERROR;
      }
      MS_ASSERT(primT->value != nullptr);
      MS_ASSERT(primT->value.AsTranspose() != nullptr);
      std::vector<int32_t> perm = primT->value.AsTranspose()->perm;
      if (perm == kPermNCHW) {
        manager->Replace(transpose_cnode, transpose_cnode->input(1));
      }
    } else if (type == schema::PrimitiveType_Conv2D) {
      auto conv_node = node->cast<CNodePtr>();
      auto typeInput = opt::GetCNodeType(conv_node->input(kTransposeInput));
      if (typeInput != schema::PrimitiveType_Transpose) {
        continue;
      }
      auto transpose_cnode = conv_node->input(kTransposeInput)->cast<CNodePtr>();
      auto primPtr = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(transpose_cnode->input(0));
      if (primPtr == nullptr) {
        MS_LOG(ERROR) << "Transpose node of onnx need to removed which has not primitiveC";
        return RET_ERROR;
      }
      auto primT = primPtr->GetPrimitiveT();
      if (primT == nullptr) {
        MS_LOG(ERROR) << "Transpose node of onnx need to removed which has not primitiveT";
        return RET_ERROR;
      }
      MS_ASSERT(primT->value != nullptr);
      MS_ASSERT(primT->value.AsTranspose() != nullptr);
      std::vector<int32_t> perm = primT->value.AsTranspose()->perm;
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
