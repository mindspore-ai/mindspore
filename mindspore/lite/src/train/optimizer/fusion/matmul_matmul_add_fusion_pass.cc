/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/train/optimizer/fusion/matmul_matmul_add_fusion_pass.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "schema/inner/model_generated.h"
#include "tools/common/meta_graph_utils.h"
#include "src/train/optimizer/common/fusion_utils.h"
namespace {
constexpr std::string_view kFirstMatMulName = "MATMUL1";
constexpr std::string_view kSecondMatMulName = "MATMUL2";
constexpr std::string_view kAddName = "ADD";
}  // namespace
namespace mindspore {
namespace lite {
/*
 * The subgraph such as the following.
 *        any                any
 *       /   \                |
 *   matmul  matmul         matmul
 *       \   /       ---->    |
 *        add                any
 *         |
 *        any
 */
namespace {
int CalNewMatMulNode(MetaGraphT *graph, const std::unique_ptr<mindspore::schema::CNodeT> &matmul_node1,
                     const std::unique_ptr<mindspore::schema::CNodeT> &matmul_node2) {
  auto &matrix_b_1 = graph->allTensors.at(matmul_node1->inputIndex.at(opt::kInputIndexOne));
  auto &matrix_b_2 = graph->allTensors.at(matmul_node2->inputIndex.at(opt::kInputIndexOne));
  if (matrix_b_1->dims != matrix_b_2->dims) {
    MS_LOG(INFO) << "currently, matmul fusion only support the same shape tensor";
    return RET_ERROR;
  }
  if (matrix_b_1->dataType != kNumberTypeFloat32 || matrix_b_2->dataType != kNumberTypeFloat32) {
    MS_LOG(INFO) << "only support float32 data type";
    return RET_ERROR;
  }
  auto matrix_b_1_data = reinterpret_cast<float *>(matrix_b_1->data.data());
  auto matrix_b_2_data = reinterpret_cast<float *>(matrix_b_2->data.data());
  int num_b = static_cast<int>(matrix_b_1->data.size() / sizeof(float));
  for (int j = 0; j < num_b; ++j) {
    matrix_b_1_data[j] += matrix_b_2_data[j];
  }
  return RET_OK;
}
}  // namespace
STATUS MatMulMatMulAddFusionPass::DefinePattern() {
  auto matmul_op1 = std::make_shared<PatternOp>();
  MS_CHECK_TRUE_RET(matmul_op1 != nullptr, RET_NULL_PTR);
  matmul_op1->id = kFirstMatMulName;
  matmul_op1->types = {schema::PrimitiveType_MatMulFusion};
  auto matmul_op2 = std::make_shared<PatternOp>();
  MS_CHECK_TRUE_RET(matmul_op2 != nullptr, RET_NULL_PTR);
  matmul_op2->id = kSecondMatMulName;
  matmul_op2->types = {schema::PrimitiveType_MatMulFusion};
  auto add_op = std::make_shared<PatternOp>();
  MS_CHECK_TRUE_RET(add_op != nullptr, RET_NULL_PTR);
  add_op->id = kAddName;
  add_op->types = {schema::PrimitiveType_AddFusion};
  add_op->left = matmul_op1;
  add_op->right = matmul_op2;
  auto fusion_pattern = std::make_unique<FusionPattern>("MatMulMatMulAddFusion");
  MS_CHECK_TRUE_MSG(fusion_pattern != nullptr, RET_NULL_PTR, "new fusion_pattern failed");
  fusion_pattern->AddPatternOp(matmul_op1);
  fusion_pattern->AddPatternOp(matmul_op2);
  fusion_pattern->AddPatternOp(add_op);
  fusion_pattern->Finish();
  this->patterns.emplace_back(fusion_pattern.release());
  return RET_OK;
}

STATUS MatMulMatMulAddFusionPass::DoFusion(MetaGraphT *graph, const std::string &pattern_name,
                                           const std::unordered_map<std::string, std::shared_ptr<Path>> &matched_path) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_NULL_PTR);
  if (matched_path.size() != opt::kMatchPathLenThree) {
    MS_LOG(INFO) << "MatMul-MatMul-Add-Fusion should have three NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }

  size_t matmul_index1 = 0;
  auto ret = opt::GetMatchNodeIndex(graph, matched_path, std::string(kFirstMatMulName), &matmul_index1);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "cannot get matmul_index1");
  auto &matmul_node1 = graph->nodes.at(matmul_index1);
  MS_CHECK_TRUE_MSG(matmul_node1 != nullptr, RET_NULL_PTR, "matmul_node1 is nullptr");
  size_t matmul_index2 = 0;
  ret = opt::GetMatchNodeIndex(graph, matched_path, std::string(kSecondMatMulName), &matmul_index2);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "cannot get matmul_index2");
  auto &matmul_node2 = graph->nodes.at(matmul_index2);
  MS_CHECK_TRUE_MSG(matmul_node2 != nullptr, RET_NULL_PTR, "matmul_node2 is nullptr");
  MS_CHECK_TRUE_MSG(matmul_node1->inputIndex.size() > C1NUM && matmul_node2->inputIndex.size() > C1NUM,
                    RET_PARAM_INVALID, "matmul should have two input at least");
  if (matmul_node1->inputIndex.size() < matmul_node2->inputIndex.size()) {
    matmul_node1.swap(matmul_node2);
  }
  size_t add_index = 0;
  ret = opt::GetMatchNodeIndex(graph, matched_path, std::string(kAddName), &add_index);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "cannot get add_index");
  auto &add_node = graph->nodes.at(add_index);
  MS_CHECK_TRUE_MSG(add_node != nullptr, RET_NULL_PTR, "add_node is nullptr");

  if (matmul_node1->quantType == schema::QuantType_QUANT_ALL ||
      matmul_node1->quantType == schema::QuantType_QUANT_DYNAMIC ||
      matmul_node2->quantType == schema::QuantType_QUANT_ALL ||
      matmul_node2->quantType == schema::QuantType_QUANT_DYNAMIC ||
      add_node->quantType == schema::QuantType_QUANT_ALL || add_node->quantType == schema::QuantType_QUANT_DYNAMIC) {
    MS_LOG(DEBUG) << "cannot fusion with quant node";
    return RET_NO_CHANGE;
  }
  MS_CHECK_TRUE_RET(matmul_node1->primitive != nullptr, RET_NULL_PTR);
  auto matmul_type1 = matmul_node1->primitive->value.AsMatMulFusion()->activation_type;
  MS_CHECK_TRUE_RET(matmul_node2->primitive != nullptr, RET_NULL_PTR);
  auto matmul_type2 = matmul_node2->primitive->value.AsMatMulFusion()->activation_type;
  MS_CHECK_TRUE_RET(add_node->primitive != nullptr, RET_NULL_PTR);
  auto add_type = add_node->primitive->value.AsAddFusion()->activation_type;
  MS_CHECK_TRUE_RET(matmul_type1 == ActivationType::ActivationType_NO_ACTIVATION &&
                      matmul_type2 == ActivationType::ActivationType_NO_ACTIVATION &&
                      add_type == ActivationType::ActivationType_NO_ACTIVATION,
                    RET_NO_CHANGE);

  if (matmul_node1->inputIndex.at(FIRST_INPUT) != matmul_node2->inputIndex.at(FIRST_INPUT)) {
    MS_LOG(INFO) << "matmul should have the same first input";
    return RET_NO_CHANGE;
  }
  auto &matmul_left_b = graph->allTensors[matmul_node1->inputIndex.at(SECOND_INPUT)];
  auto &matmul_right_b = graph->allTensors[matmul_node2->inputIndex.at(SECOND_INPUT)];
  if (matmul_left_b->data.empty() || matmul_right_b->data.empty()) {
    return RET_NO_CHANGE;
  }
  if (CalNewMatMulNode(graph, matmul_node1, matmul_node2) != RET_OK) {
    MS_LOG(INFO) << "failed to fusion two matmul";
    return RET_NO_CHANGE;
  }

  matmul_node1->outputIndex = {add_node->outputIndex};
  // cannot delete node here, otherwise will destroy order in other pattern's node index
  // make it an isolated node to be removed in IsolatedNodeRemovePass
  matmul_node2->inputIndex.clear();
  matmul_node2->outputIndex.clear();
  add_node->inputIndex.clear();
  add_node->outputIndex.clear();
  return RET_OK;
}

MatMulMatMulAddFusionPass::~MatMulMatMulAddFusionPass() = default;
}  // namespace lite
}  // namespace mindspore
