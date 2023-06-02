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
#include "tools/optimizer/fusion/tensor_dot_fusion.h"
#include <memory>
#include <vector>
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "ops/op_utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
STATUS GetIndexValue(const CNodePtr &cnode, std::vector<int> *index, int node_index) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(index != nullptr);
  if (utils::isa<CNodePtr>(cnode->input(node_index))) {
    return RET_ERROR;
  }
  lite::DataInfo data_info;
  int status = RET_ERROR;
  if (utils::isa<ParameterPtr>(cnode->input(node_index))) {
    status = lite::FetchDataFromParameterNode(cnode, node_index, converter::kFmkTypeMs, &data_info, true);
  } else {
    status = lite::FetchDataFromValueNode(cnode, node_index, converter::kFmkTypeMs, false, &data_info, true);
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "fetch gather index data failed.";
    return RET_ERROR;
  }
  if ((data_info.data_type_ != kNumberTypeInt32 && data_info.data_type_ != kNumberTypeInt) ||
      data_info.shape_.size() != 1) {
    MS_LOG(ERROR) << "gather index data is invalid.";
    return RET_ERROR;
  }
  index->resize(data_info.shape_[0]);
  if (!data_info.data_.empty() &&
      memcpy_s(index->data(), index->size() * sizeof(int), data_info.data_.data(), data_info.data_.size()) != EOK) {
    MS_LOG(ERROR) << "copy index data failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

const BaseRef TensorDotFusion::DefinePattern() const {
  // match tf tensordot operator
  auto is_var = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var != nullptr, {});
  auto is_shape = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimShape>);
  MS_CHECK_TRUE_RET(is_shape != nullptr, {});
  auto shape = VectorRef({is_shape, is_var});

  auto is_gather_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimGather>);
  MS_CHECK_TRUE_RET(is_gather_1 != nullptr, {});
  auto is_param_1 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_1 != nullptr, {});
  auto is_param_2 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_2 != nullptr, {});
  auto gather_1 = VectorRef({is_gather_1, shape, is_param_1, is_param_2});

  auto is_gather_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimGather>);
  MS_CHECK_TRUE_RET(is_gather_2 != nullptr, {});
  auto is_param_3 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_3 != nullptr, {});
  auto is_param_4 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_4 != nullptr, {});
  auto gather_2 = VectorRef({is_gather_2, shape, is_param_3, is_param_4});

  auto is_reduce_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReduceFusion>);
  MS_CHECK_TRUE_RET(is_reduce_1 != nullptr, {});
  auto is_param_5 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_5 != nullptr, {});
  auto reduce_1 = VectorRef({is_reduce_1, gather_2, is_param_5});

  auto is_reduce_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReduceFusion>);
  MS_CHECK_TRUE_RET(is_reduce_2 != nullptr, {});
  auto is_param_6 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_6 != nullptr, {});
  auto reduce_2 = VectorRef({is_reduce_2, gather_1, is_param_6});

  auto is_statck = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStack>);
  MS_CHECK_TRUE_RET(is_statck != nullptr, {});
  auto stack = VectorRef({is_statck, reduce_2, reduce_1});

  auto is_trans = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_trans != nullptr, {});
  auto is_param_7 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_7 != nullptr, {});
  auto trans = VectorRef({is_trans, is_var, is_param_7});

  auto is_reshape_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape_1 != nullptr, {});
  auto reshape_1 = VectorRef({is_reshape_1, trans, stack});

  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto is_param_8 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_8 != nullptr, {});
  auto matmul = VectorRef({is_matmul, reshape_1, is_param_8});

  auto is_concat = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimConcat>);
  MS_CHECK_TRUE_RET(is_concat != nullptr, {});
  auto is_param_9 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_9 != nullptr, {});
  auto concat = VectorRef({is_concat, gather_1, is_param_9});

  auto is_reshape_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape_2 != nullptr, {});
  auto reshape_2 = VectorRef({is_reshape_2, matmul, concat});
  return reshape_2;
}

const AnfNodePtr TensorDotFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }
  auto reshape_1_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_1_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(reshape_1_cnode->input(1) != nullptr, nullptr);
  auto matmul_cnode = reshape_1_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(reshape_1_cnode->input(kInputIndexTwo) != nullptr, nullptr);
  auto concat_cnode = reshape_1_cnode->input(kInputIndexTwo)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(matmul_cnode->input(1) != nullptr, nullptr);
  auto reshape_2_cnode = matmul_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_2_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(reshape_2_cnode->input(kInputIndexTwo) != nullptr, nullptr);
  auto stack_cnode = reshape_2_cnode->input(kInputIndexTwo)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(stack_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(reshape_2_cnode->input(1) != nullptr, nullptr);
  auto trans_cnode = reshape_2_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(trans_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(stack_cnode->input(kInputIndexTwo) != nullptr, nullptr);
  auto reduce_1_cnode = stack_cnode->input(kInputIndexTwo)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reduce_1_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(stack_cnode->input(1) != nullptr, nullptr);
  auto reduce_2_cnode = stack_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reduce_2_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(reduce_1_cnode->input(1) != nullptr, nullptr);
  auto gather_2_cnode = reduce_1_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(gather_2_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(reduce_2_cnode->input(1) != nullptr, nullptr);
  auto gather_1_cnode = reduce_2_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(gather_1_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(gather_1_cnode->input(1) != nullptr, nullptr);
  auto shape_cnode = gather_1_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(shape_cnode != nullptr, nullptr);

  std::vector<int> gather_1_index;
  std::vector<int> gather_2_index;
  std::vector<int> concat_value;
  auto status = GetIndexValue(gather_1_cnode, &gather_1_index, kInputIndexTwo);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get gather cnode index failed.";
    return nullptr;
  }
  status = GetIndexValue(gather_2_cnode, &gather_2_index, kInputIndexTwo);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get gather cnode index failed.";
    return nullptr;
  }
  if (utils::isa<ParameterPtr>(concat_cnode->input(kInputIndexTwo)) &&
      utils::isa<ValueNodePtr>(concat_cnode->input(kInputIndexTwo))) {
    return nullptr;
  }
  status = GetIndexValue(concat_cnode, &concat_value, kInputIndexTwo);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get gather cnode index failed.";
    return nullptr;
  }
  auto manage = Manage(func_graph);
  MS_CHECK_TRUE_RET(manage != nullptr, nullptr);
  // For special shapes, it can be directly converted into matmul operator and reshape operator.
  if (gather_1_index.size() == 1 && gather_2_index.size() == 1 && concat_value.size() == 0) {
    manage->SetEdge(matmul_cnode, 1, shape_cnode->input(1));
    manage->Replace(concat_cnode, gather_1_cnode);
    return nullptr;
  }
  // For special shapes, it can be directly converted into matmul operator
  if (gather_1_index.size() == 1 && gather_2_index.size() == 1 && concat_value.size() == 1) {
    manage->SetEdge(matmul_cnode, 1, shape_cnode->input(1));
    manage->Replace(reshape_1_cnode, matmul_cnode);
    return nullptr;
  }
  return nullptr;
}
}  // namespace mindspore::opt
