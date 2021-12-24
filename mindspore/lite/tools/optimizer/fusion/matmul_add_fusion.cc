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

#include "tools/optimizer/fusion/matmul_add_fusion.h"
#include <vector>
#include <memory>
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
bool CheckAndGetMatMulIndex(const CNodePtr &cnode, size_t *index) {
  MS_ASSERT(cnode != nullptr && index != nullptr);
  if (cnode->size() != kInputSizeThree) {
    return false;
  }
  size_t matmul_index = 0;
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (CheckPrimitiveType(cnode->input(i), prim::kPrimMatMulFusion)) {
      matmul_index = i;
      break;
    }
  }
  if (matmul_index == 0) {
    return false;
  }
  *index = matmul_index;
  return true;
}
bool IsPrimitiveProper(const CNodePtr &add_cnode, const CNodePtr &matmul_cnode, int index) {
  auto add_primc = GetValueNode<std::shared_ptr<ops::AddFusion>>(add_cnode->input(0));
  MS_CHECK_TRUE_RET(add_primc != nullptr, false);
  if (IsQuantParameterNode(add_primc)) {
    MS_LOG(INFO) << add_cnode->fullname_with_scope() << " is quant node";
    return false;
  }

  auto bias_node = add_cnode->input(kInputSizeThree - index);
  if (!utils::isa<ValueNode>(bias_node) &&
      (!utils::isa<Parameter>(bias_node) || !bias_node->cast<ParameterPtr>()->default_param())) {
    return false;
  }
  auto abstract = bias_node->abstract();
  MS_CHECK_TRUE_RET(abstract != nullptr, false);
  std::vector<int64_t> bias_shape;
  if (FetchShapeFromAbstract(abstract, &bias_shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Fetch shape from abstract failed.";
    return false;
  }

  if (bias_shape.size() > DIMENSION_1D) {
    MS_LOG(INFO) << "only support bias with shape size of 1.";
    return false;
  }

  if (matmul_cnode->size() > kInputSizeThree) {
    auto matmul_bias_node = matmul_cnode->input(kInputIndexThree);
    if (!IsParamNode(matmul_bias_node)) {
      MS_LOG(INFO) << matmul_cnode->fullname_with_scope() << "'s bias is not parameter";
      return false;
    }
  }

  auto matmul_primc = GetValueNode<std::shared_ptr<ops::MatMulFusion>>(matmul_cnode->input(0));
  MS_CHECK_TRUE_RET(matmul_primc != nullptr, false);
  if (matmul_primc->GetAttr(ops::kActivationType) != nullptr &&
      matmul_primc->get_activation_type() != ActivationType::NO_ACTIVATION) {
    MS_LOG(INFO) << matmul_cnode->fullname_with_scope() << " has activation attr";
    return false;
  }
  if (IsQuantParameterNode(matmul_primc)) {
    MS_LOG(INFO) << matmul_cnode->fullname_with_scope() << "is quant node";
    return false;
  }

  return true;
}

int CalNewCnodeBias(const AnfNodePtr &add_weight_node, const CNodePtr &matmul_cnode) {
  MS_CHECK_TRUE_RET(add_weight_node != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, RET_ERROR);
  auto matmul_bias_node = matmul_cnode->input(kInputIndexThree);
  MS_CHECK_TRUE_RET(matmul_bias_node != nullptr, RET_ERROR);
  std::shared_ptr<tensor::Tensor> matmul_bias_tensor = GetTensorInfo(matmul_bias_node);
  MS_CHECK_TRUE_RET(matmul_bias_tensor != nullptr, RET_ERROR);
  if (matmul_bias_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(INFO) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> matmul_bias_shape = matmul_bias_tensor->shape();
  auto matmul_bias_data = reinterpret_cast<float *>(matmul_bias_tensor->data_c());
  MS_CHECK_TRUE_RET(matmul_bias_data != nullptr, RET_ERROR);

  std::shared_ptr<tensor::Tensor> add_weight_tensor = GetTensorInfo(add_weight_node);
  MS_CHECK_TRUE_RET(add_weight_tensor != nullptr, RET_ERROR);
  if (add_weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(INFO) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> add_weight_shape = add_weight_tensor->shape();
  MS_CHECK_TRUE_RET(matmul_bias_shape == add_weight_shape, RET_ERROR);
  auto add_weight_data = reinterpret_cast<float *>(add_weight_tensor->data_c());
  MS_CHECK_TRUE_RET(add_weight_data != nullptr, RET_ERROR);

  for (int64_t i = 0; i < matmul_bias_shape[0]; ++i) {
    matmul_bias_data[i] += add_weight_data[i];
  }
  return RET_OK;
}
}  // namespace

bool MatMulAddFusion::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, false);
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!CheckPrimitiveType(node, prim::kPrimAddFusion) && !CheckPrimitiveType(node, prim::kPrimBiasAdd)) {
      continue;
    }
    if (IsMarkedTrainOp(cnode)) {
      return false;
    }
    size_t index = 0;
    if (!CheckAndGetMatMulIndex(cnode, &index)) {
      continue;
    }
    auto matmul_cnode = cnode->input(index)->cast<CNodePtr>();
    MS_ASSERT(matmul_cnode != nullptr);
    if (IsMarkedTrainOp(matmul_cnode)) {
      return false;
    }

    if (IsMultiOutputTensors(func_graph, matmul_cnode)) {
      return false;
    }

    if (!IsPrimitiveProper(cnode, matmul_cnode, index)) {
      continue;
    }

    auto manager = func_graph->manager();
    auto bias_node = cnode->input(kInputSizeThree - index);
    MS_CHECK_TRUE_RET(manager != nullptr, false);
    if (matmul_cnode->size() == kInputSizeThree) {
      manager->AddEdge(matmul_cnode, bias_node);
    } else if (matmul_cnode->size() == kInputSizeFour) {
      auto matmul_bias_node = matmul_cnode->input(kInputSizeThree);
      if (!IsParamNode(matmul_bias_node)) {
        MS_LOG(INFO) << matmul_bias_node->fullname_with_scope() << "'s bias is not parameter";
        return false;
      }

      if (CalNewCnodeBias(bias_node, matmul_cnode) != RET_OK) {
        MS_LOG(INFO) << cnode->fullname_with_scope() << " failed to fusion with "
                     << matmul_cnode->fullname_with_scope();
        return false;
      }
    }
    auto add_primc = GetValueNode<std::shared_ptr<ops::AddFusion>>(cnode->input(0));
    MS_CHECK_TRUE_RET(add_primc != nullptr, false);
    auto matmul_primc = GetValueNode<std::shared_ptr<ops::MatMulFusion>>(matmul_cnode->input(0));
    MS_CHECK_TRUE_RET(matmul_primc != nullptr, false);
    if (add_primc->GetAttr(ops::kActivationType) != nullptr &&
        add_primc->get_activation_type() != ActivationType::NO_ACTIVATION) {
      matmul_primc->set_activation_type(add_primc->get_activation_type());
    }
    matmul_cnode->set_fullname_with_scope(node->fullname_with_scope());
    (void)manager->Replace(node, matmul_cnode);
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
