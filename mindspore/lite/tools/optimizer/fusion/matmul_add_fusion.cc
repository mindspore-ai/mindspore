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
#include "tools/optimizer/fusion/matmul_add_fusion.h"
#include <vector>
#include <memory>
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
namespace {
bool IsPrimitiveProper(const CNodePtr &add_cnode, const CNodePtr &matmul_cnode, int index) {
  auto add_primc = GetValueNode<PrimitiveCPtr>(add_cnode->input(0));
  MS_CHECK_TRUE_RET(add_primc != nullptr, false);
  if (IsQuantParameterNode(add_primc)) {
    MS_LOG(INFO) << add_cnode->fullname_with_scope() << " is quant node";
    return false;
  }

  auto add_param_node = add_cnode->input(kInputSizeThree - index);
  if (!utils::isa<ValueNode>(add_param_node) &&
      (!utils::isa<Parameter>(add_param_node) || !add_param_node->cast<ParameterPtr>()->default_param())) {
    return false;
  }
  auto abstract = add_param_node->abstract();
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
    if (!utils::isa<ValueNode>(matmul_bias_node) &&
        (!utils::isa<Parameter>(matmul_bias_node) || !matmul_bias_node->cast<ParameterPtr>()->default_param())) {
      MS_LOG(INFO) << matmul_cnode->fullname_with_scope() << "'s bias is not parameter";
      return false;
    }
  }
  auto matmul_primc = ops::GetOperator<ops::MatMulFusion>(matmul_cnode->input(0));
  MS_CHECK_TRUE_RET(matmul_primc != nullptr, false);
  if (matmul_primc->GetAttr(ops::kActivationType) != nullptr &&
      matmul_primc->get_activation_type() != ActivationType::NO_ACTIVATION) {
    MS_LOG(INFO) << matmul_cnode->fullname_with_scope() << " has activation attr";
    return false;
  }
  auto matmul_prim_c = matmul_primc->GetPrim();
  MS_CHECK_TRUE_RET(matmul_prim_c != nullptr, false);
  if (IsQuantParameterNode(matmul_prim_c)) {
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
    auto add_cnode = node->cast<CNodePtr>();
    if (!CheckPrimitiveType(node, prim::kPrimAddFusion) && !CheckPrimitiveType(node, prim::kPrimBiasAdd)) {
      continue;
    }
    if (IsMarkedTrainOp(add_cnode)) {
      continue;
    }
    size_t index = 0;

    if (!CheckAndGetCnodeIndex(add_cnode, &index, prim::kPrimMatMulFusion)) {
      continue;
    }

    auto matmul_cnode = add_cnode->input(index)->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(matmul_cnode != nullptr, false);
    if (IsMarkedTrainOp(matmul_cnode)) {
      continue;
    }

    if (IsMultiOutputTensors(func_graph, matmul_cnode)) {
      continue;
    }

    if (!IsPrimitiveProper(add_cnode, matmul_cnode, index)) {
      continue;
    }
    auto manager = func_graph->manager();
    MS_CHECK_TRUE_RET(manager != nullptr, false);
    auto add_param_node = add_cnode->input(kInputSizeThree - index);
    if (matmul_cnode->size() == kInputSizeThree) {
      manager->AddEdge(matmul_cnode, add_param_node);
    } else if (matmul_cnode->size() == kInputSizeFour) {
      if (CalNewCnodeBias(add_param_node, matmul_cnode) != RET_OK) {
        MS_LOG(INFO) << add_cnode->fullname_with_scope() << " failed to fusion with "
                     << matmul_cnode->fullname_with_scope();
        return false;
      }
    }

    if (CheckPrimitiveType(node, prim::kPrimAddFusion)) {
      auto add_primc = ops::GetOperator<ops::AddFusion>(add_cnode->input(0));
      MS_CHECK_TRUE_RET(add_primc != nullptr, false);
      if (add_primc->GetAttr(ops::kActivationType) != nullptr &&
          add_primc->get_activation_type() != ActivationType::NO_ACTIVATION) {
        auto matmul_primc = ops::GetOperator<ops::MatMulFusion>(matmul_cnode->input(0));
        MS_CHECK_TRUE_RET(matmul_primc != nullptr, false);
        matmul_primc->set_activation_type(add_primc->get_activation_type());
      }
    }
    (void)manager->Replace(node, matmul_cnode);
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
