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
#include <memory>
#include "tools/optimizer/graph/eliminate_redundant_cast_pass.h"
#include "tools/optimizer/graph/infershape_pass.h"

namespace mindspore::opt {
int EliminateRedundantCastPass::RemoveCastOp(const AnfNodePtr &anf_node, const FuncGraphManagerPtr &manager) {
  const int expected_cast_input_count = 3;
  auto cast_cnode = anf_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cast_cnode->inputs().size() == expected_cast_input_count, lite::RET_NO_CHANGE);
  TypeId first_type;
  TypeId second_type;
  if (opt::GetDataTypeFromAnfNode(cast_cnode->input(1), &first_type) != RET_OK) {
    MS_LOG(ERROR) << "Failed to get " << anf_node->fullname_with_scope() << " output tensor data type.";
    return lite::RET_NO_CHANGE;
  }

  auto dst_type_tensor = cast_cnode->input(2)->cast<ParameterPtr>();
  MS_CHECK_TRUE_RET(dst_type_tensor != nullptr, lite::RET_NO_CHANGE);
  auto tensor_info = std::dynamic_pointer_cast<tensor::Tensor>(dst_type_tensor->default_param());
  MS_CHECK_TRUE_RET(tensor_info != nullptr, lite::RET_NO_CHANGE);
  MS_CHECK_TRUE_RET(tensor_info->ElementsNum() == 1, lite::RET_NO_CHANGE);

  second_type = static_cast<TypeId>(static_cast<int *>(tensor_info->data_c())[0]);
  if (first_type == second_type) {
    MS_LOG(DEBUG) << "Cast node " << anf_node->fullname_with_scope() << " is eliminated.";
    (void)this->remove_cnode_.insert(anf_node);
    return manager->Replace(anf_node, cast_cnode->input(1)) ? RET_OK : RET_ERROR;
  }
  return lite::RET_NO_CHANGE;
}

bool EliminateRedundantCastPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto infer_shape_pass = std::make_shared<InferShapePass>(this->fmk_type_, this->train_flag_, true);
  if (!infer_shape_pass->Run(func_graph)) {
    return true;
  }
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  auto node_list = TopoSort(func_graph->get_return());
  int status = RET_OK;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimCast)) {
      status = this->RemoveCastOp(node, manager);
    }
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Failed to run cast elimination pass.";
      return false;
    }
  }
  for (auto &node : this->remove_cnode_) {
    func_graph->DropNode(node);
  }
  return true;
}
}  // namespace mindspore::opt
