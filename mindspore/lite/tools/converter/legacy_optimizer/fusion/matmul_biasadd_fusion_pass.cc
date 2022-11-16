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

#include "tools/converter/legacy_optimizer/fusion/matmul_biasadd_fusion_pass.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "schema/inner/model_generated.h"
#include "tools/common/meta_graph_utils.h"
namespace {
constexpr int kNumBiasMatchPathLen = 2;
constexpr std::string_view MulName = "MATMUL";
constexpr std::string_view BiasName = "BIASADD";
}  // namespace
namespace mindspore {
namespace lite {
STATUS MatMulBiasAddFusionPass::Run(MetaGraphT *graph) { return FusionPass::Run(graph); }

STATUS MatMulBiasAddFusionPass::DefinePattern() {
  auto mul_op = std::make_shared<PatternOp>();
  mul_op->id = MulName;
  mul_op->types = {schema::PrimitiveType_MatMulFusion};
  auto bias_op = std::make_shared<PatternOp>();
  bias_op->id = BiasName;
  bias_op->types = {schema::PrimitiveType_BiasAdd};
  bias_op->left = mul_op;
  std::unique_ptr<FusionPattern> fusion_pattern(new (std::nothrow) FusionPattern("MatMulBiasAddFusion"));
  if (fusion_pattern == nullptr) {
    MS_LOG(ERROR) << "new fusion_pattern failed";
    return RET_ERROR;
  }
  fusion_pattern->AddPatternOp(mul_op);
  fusion_pattern->AddPatternOp(bias_op);
  fusion_pattern->Finish();
  this->patterns.emplace_back(fusion_pattern.release());
  return RET_OK;
}

STATUS MatMulBiasAddFusionPass::DoFusion(MetaGraphT *graph, const std::string &pattern_name,
                                         const std::unordered_map<std::string, std::shared_ptr<Path>> &matched_path) {
  MS_ASSERT(graph != nullptr);
  if (matched_path.size() != kNumBiasMatchPathLen) {
    MS_LOG(ERROR) << "MatMul-BiasAdd-Fusion should have two NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }
  auto mul_path_iter = matched_path.find(std::string(MulName));
  MS_ASSERT(mul_path_iter != matched_path.end());
  auto &mul_path = mul_path_iter->second;
  MS_ASSERT(mul_path != nullptr);
  auto bias_path_iter = matched_path.find(std::string(BiasName));
  MS_ASSERT(bias_path_iter != matched_path.end());
  auto &bias_path = bias_path_iter->second;
  MS_ASSERT(bias_path != nullptr);
  auto mul_index = mul_path->nodeIdx;
  auto bias_index = bias_path->nodeIdx;
  auto &mul_node = graph->nodes.at(mul_index);
  MS_ASSERT(mul_node != nullptr);
  auto &bias_node = graph->nodes.at(bias_index);
  MS_ASSERT(bias_node != nullptr);
  auto bias_tensor_index = bias_node->inputIndex.at(1);
  if (mul_node->inputIndex.size() != 2) {
    MS_LOG(DEBUG) << "cat not fusion.";
    return RET_NO_CHANGE;
  }
  mul_node->inputIndex.push_back(bias_tensor_index);
  mul_node->outputIndex = {bias_node->outputIndex};
  // cannot delete node here, otherwise will destroy order in other pattern's node index
  // make it an isolated node to be removed in IsolatedNodeRemovePass
  bias_node->inputIndex.clear();
  bias_node->outputIndex.clear();
  return RET_OK;
}

MatMulBiasAddFusionPass::~MatMulBiasAddFusionPass() = default;
}  // namespace lite
}  // namespace mindspore
