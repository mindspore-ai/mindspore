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

#include "src/train/optimizer/fusion/matmul_add_fusion_pass.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "schema/inner/model_generated.h"
#include "tools/common/meta_graph_utils.h"
#include "src/train/optimizer/common/fusion_utils.h"

namespace {
constexpr int kNumAddMatchPathLen = 2;
constexpr std::string_view MatMulName = "MATMUL";
constexpr std::string_view AddName = "ADD";
}  // namespace
namespace mindspore {
namespace lite {
namespace {
int CalNewCnodeBias(const std::unique_ptr<mindspore::schema::TensorT> &add_weight_tensor,
                    const std::unique_ptr<mindspore::schema::TensorT> &matmul_bias_tensor) {
  if (add_weight_tensor->dataType != kNumberTypeFloat32 || matmul_bias_tensor->dataType != kNumberTypeFloat32) {
    MS_LOG(INFO) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int32_t> matmul_bias_shape = matmul_bias_tensor->dims;
  std::vector<int32_t> add_weight_shape = add_weight_tensor->dims;
  MS_CHECK_TRUE_RET(matmul_bias_shape == add_weight_shape, RET_ERROR);
  auto add_weight_data = reinterpret_cast<float *>(add_weight_tensor->data.data());
  auto matmul_bias_data = reinterpret_cast<float *>(matmul_bias_tensor->data.data());
  int num = static_cast<int>(matmul_bias_tensor->data.size() / sizeof(float));
  for (int i = 0; i < num; ++i) {
    matmul_bias_data[i] += add_weight_data[i];
  }
  return RET_OK;
}
}  // namespace
STATUS MatMulAddFusionPass::Run(MetaGraphT *graph) { return FusionPass::Run(graph); }

STATUS MatMulAddFusionPass::DefinePattern() {
  auto matmul_op = std::make_shared<PatternOp>();
  CHECK_NULL_RETURN(matmul_op);
  matmul_op->id = MatMulName;
  matmul_op->types = {schema::PrimitiveType_MatMulFusion};
  auto add_op = std::make_shared<PatternOp>();
  CHECK_NULL_RETURN(add_op);
  add_op->id = AddName;
  add_op->types = {schema::PrimitiveType_AddFusion};
  add_op->left = matmul_op;
  std::unique_ptr<FusionPattern> fusion_pattern(new (std::nothrow) FusionPattern("MatMulAddFusion"));
  if (fusion_pattern == nullptr) {
    MS_LOG(ERROR) << "new fusion_pattern failed";
    return RET_ERROR;
  }
  fusion_pattern->AddPatternOp(matmul_op);
  fusion_pattern->AddPatternOp(add_op);
  fusion_pattern->Finish();
  this->patterns.emplace_back(fusion_pattern.release());
  return RET_OK;
}

STATUS MatMulAddFusionPass::DoFusion(MetaGraphT *graph, const std::string &pattern_name,
                                     const std::unordered_map<std::string, std::shared_ptr<Path>> &matched_path) {
  MSLITE_CHECK_PTR(graph);
  if (matched_path.size() != kNumAddMatchPathLen) {
    MS_LOG(ERROR) << "MatMul-Add-Fusion should have two NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }
  auto matmul_path_iter = matched_path.find(std::string(MatMulName));
  MS_CHECK_TRUE_RET(matmul_path_iter != matched_path.end(), RET_NO_CHANGE);
  auto &matmul_path = matmul_path_iter->second;
  MSLITE_CHECK_PTR(matmul_path);
  auto add_path_iter = matched_path.find(std::string(AddName));
  MS_CHECK_TRUE_RET(add_path_iter != matched_path.end(), RET_NO_CHANGE);
  auto &add_path = add_path_iter->second;
  MSLITE_CHECK_PTR(add_path);
  auto matmul_index = matmul_path->nodeIdx;
  auto add_index = add_path->nodeIdx;
  auto &matmul_node = graph->nodes.at(matmul_index);
  MSLITE_CHECK_PTR(matmul_node);
  auto &add_node = graph->nodes.at(add_index);
  MSLITE_CHECK_PTR(add_node);
  if (matmul_node->quantType == schema::QuantType_QUANT_ALL ||
      matmul_node->quantType == schema::QuantType_QUANT_DYNAMIC ||
      matmul_node->outputIndex.size() != opt::kOutputSizeOne || add_node->quantType == schema::QuantType_QUANT_ALL ||
      add_node->quantType == schema::QuantType_QUANT_DYNAMIC) {
    MS_LOG(DEBUG) << "cannot fusion.";
    return RET_NO_CHANGE;
  }
  MS_CHECK_TRUE_RET(matmul_node->primitive != nullptr, RET_NULL_PTR);
  auto matmul_type = matmul_node->primitive->value.AsMatMulFusion();
  MS_CHECK_TRUE_RET(matmul_type->activation_type == ActivationType::ActivationType_NO_ACTIVATION, RET_NO_CHANGE);
  auto add_param_shape = graph->allTensors.at(add_node->inputIndex.at(SECOND_INPUT))->dims;
  MS_CHECK_TRUE_MSG(add_param_shape.size() == DIMENSION_1D, RET_NO_CHANGE, "only support bias with shape size of 1.");
  if (matmul_node->inputIndex.size() == C3NUM) {
    auto &matmul_bias_tensor = graph->allTensors.at(matmul_node->inputIndex.at(THIRD_INPUT));
    if (matmul_bias_tensor->data.data() == nullptr) {
      MS_LOG(INFO) << matmul_node->name << "'s bias is not const";
      return RET_NO_CHANGE;
    }
    auto &add_weight_tensor = graph->allTensors.at(add_node->inputIndex.at(SECOND_INPUT));
    if (CalNewCnodeBias(add_weight_tensor, matmul_bias_tensor) != RET_OK) {
      MS_LOG(INFO) << add_node->name << " failed to fusion with " << matmul_node->name;
      return RET_NO_CHANGE;
    }
  }

  auto add_tensor_index = add_node->inputIndex.at(SECOND_INPUT);
  if (matmul_node->inputIndex.size() == C2NUM) {
    matmul_node->inputIndex.push_back(add_tensor_index);
  }
  matmul_node->outputIndex = {add_node->outputIndex};
  // cannot delete node here, otherwise will destroy order in other pattern's node index
  // make it an isolated node to be removed in IsolatedNodeRemovePass
  add_node->inputIndex.clear();
  add_node->outputIndex.clear();
  return RET_OK;
}

MatMulAddFusionPass::~MatMulAddFusionPass() = default;
}  // namespace lite
}  // namespace mindspore
