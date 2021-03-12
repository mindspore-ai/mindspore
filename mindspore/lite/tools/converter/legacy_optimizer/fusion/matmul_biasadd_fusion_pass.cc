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

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <memory>
#include "tools/converter/legacy_optimizer/fusion/matmul_biasadd_fusion_pass.h"
#include "src/common/log_adapter.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
#define MATMUL_BIASADD_MATCH_PATH_LEN 2
#define BIASADD_OP_BIAS_INDEX 1
#define BIASADD_OP_INPUT_NUM 2

STATUS MatMulBiasAddFusionPass::Run(MetaGraphT *graph) { return FusionPass::Run(graph); }

STATUS MatMulBiasAddFusionPass::DefinePattern() {
  auto matMulOp = std::make_shared<PatternOp>();
  matMulOp->id = MATMUL_NAME;
  matMulOp->types = {schema::PrimitiveType_MatMul};
  auto baOp = std::make_shared<PatternOp>();
  baOp->id = BIASADD_NAME;
  baOp->types = {schema::PrimitiveType_BiasAdd};
  baOp->left = matMulOp;

  std::unique_ptr<FusionPattern> fusionPattern(new (std::nothrow) FusionPattern("MatMulBiasAddFusion"));
  if (fusionPattern == nullptr) {
    MS_LOG(ERROR) << "new fusionPattern failed";
    return RET_ERROR;
  }
  fusionPattern->AddPatternOp(matMulOp);
  fusionPattern->AddPatternOp(baOp);
  fusionPattern->Finish();

  this->patterns.emplace_back(fusionPattern.release());

  return RET_OK;
}

STATUS MatMulBiasAddFusionPass::DoFusion(MetaGraphT *graph, const std::string &patternName,
                                         std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  if (matchedPath.size() != MATMUL_BIASADD_MATCH_PATH_LEN) {
    MS_LOG(ERROR) << "MatMul-BiasAdd-Fusion should have two NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }

  auto matMulPath = matchedPath[MATMUL_NAME];
  auto baPath = matchedPath[BIASADD_NAME];
  auto &matMulNode = graph->nodes.at(matMulPath->nodeIdx);
  auto &baNode = graph->nodes.at(baPath->nodeIdx);
  // can not check shape because there is now shape infer in converter
  MS_ASSERT(matMulNode != nullptr);
  MS_ASSERT(matMulNode->inputIndex.size() == 2);
  // biasadd node the second tensor is not constant tensor, don't fusion
  auto baNodeInputIndex = baNode->inputIndex;
  if (baNodeInputIndex.size() != BIASADD_OP_INPUT_NUM) {
    MS_LOG(ERROR) << "input num is invalid! node: " << baNode->name.c_str();
    return RET_ERROR;
  }
  MS_ASSERT(graph->allTensors.size() > baNodeInputIndex.at(BIASADD_OP_BIAS_INDEX));
  const auto &baNodeBiasTensor = graph->allTensors.at(baNodeInputIndex.at(BIASADD_OP_BIAS_INDEX));
  MS_ASSERT(baNodeBiasTensor != nullptr);
  if (baNodeBiasTensor->refCount != NodeType_ValueNode) {
    // dont fusion, return
    return RET_OK;
  }

  // 1. add biasTensor for matMul
  auto status = AddFullConnectionBiasTensor(matMulPath, baPath, graph);
  if (RET_OK != status) {
    MS_LOG(ERROR) << "AddFullConnectionBiasTensor failed, ret: " << status;
    return status;
  }

  // 2. change matmul to full connection op
  matMulNode->name += "-fc";
  std::unique_ptr<FullConnectionT> fcAttr(new (std::nothrow) FullConnectionT());
  if (fcAttr == nullptr) {
    MS_LOG(ERROR) << "new FullConnectionT node failed";
    return RET_ERROR;
  }
  fcAttr->has_bias = true;
  fcAttr->axis = 1;
  MS_ASSERT(matMulNode->primitive != nullptr);
  MS_ASSERT(matMulNode->primitive->value != nullptr);
  MS_ASSERT(matMulNode->primitive->value.AsMatMul() != nullptr);
  transA = matMulNode->primitive->value.AsMatMul()->transpose_a;
  transB = matMulNode->primitive->value.AsMatMul()->transpose_b;
  matMulNode->primitive->value.type = schema::PrimitiveType_FullConnection;
  matMulNode->primitive->value.value = fcAttr.release();

  // 3. delete BiasAdd node
  MergeNodeAttrFromPost(matMulNode, baNode);
  status = IsolateOneWayNode(graph, baPath->nodeIdx);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode failed, subGraph: " << baPath->subGraphIdx << ", node: " << baPath->nodeIdx
                  << ", ret: " << status;
    return status;
  }

  // 4. addTranspose node
  status = InsertTransposeNode(graph, matMulPath);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "InsertTransposeNode failed, subGraph: " << matMulPath->subGraphIdx
                  << ", node: " << matMulPath->nodeIdx << ", ret: " << status;
    return status;
  }
  return RET_OK;
}

STATUS MatMulBiasAddFusionPass::InsertTransposeNode(MetaGraphT *graph, const std::shared_ptr<Path> &matMulPath) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(matMulPath != nullptr);

  std::vector<size_t> insertNodeIdxList;
  if (transA) {
    insertNodeIdxList.emplace_back(0);
  }
  if (!transB) {
    insertNodeIdxList.emplace_back(1);
  }

  auto matmulOpIter = graph->nodes.begin() + matMulPath->nodeIdx;
  STATUS errorCode = RET_OK;
  auto perm_tensor = std::make_unique<schema::TensorT>();
  perm_tensor->dataType = kNumberTypeInt32;
  perm_tensor->dims = {2};
  std::vector<int> perm{1, 0};
  size_t bytes = perm.size() * sizeof(int);
  perm_tensor->data.resize(bytes);
  perm_tensor->name = "perm_" + std::to_string(id++);
  if (memcpy_s(perm_tensor->data.data(), bytes, perm.data(), bytes) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    return RET_ERROR;
  }
  size_t index = graph->allTensors.size();
  graph->allTensors.push_back(std::move(perm_tensor));
  for (auto needInsertIdx : insertNodeIdxList) {
    auto transNode = std::unique_ptr<CNodeT>(new (std::nothrow) CNodeT);
    if (transNode == nullptr) {
      MS_LOG(ERROR) << "new TransNode failed";
      return RET_ERROR;
    }
    transNode->name = "transpose" + std::to_string(id++);
    transNode->primitive->value.type = schema::PrimitiveType_Transpose;
    int insert_num = 0;
    matmulOpIter = InsertNode(graph, matmulOpIter, kBefore, needInsertIdx, std::move(transNode), &errorCode,
                              &insert_num, TransposeOpCopyer);
    if (errorCode != RET_OK) {
      MS_LOG(ERROR) << "InsertNode failed: " << errorCode;
      return errorCode;
    }
    for (int i = insert_num; i > 0; --i) {
      (*(matmulOpIter - i))->inputIndex.push_back(index);
    }
  }
  graph->allTensors.at(index)->refCount = insertNodeIdxList.size();
  return RET_OK;
}

#define BIASADD_WEIGHT_SHAPE_SIZE 1
#define BIASADD_BIAS_DIM_INDEX 0

STATUS MatMulBiasAddFusionPass::AddFullConnectionBiasTensor(const std::shared_ptr<Path> &matMulPath,
                                                            const std::shared_ptr<Path> &baPath, MetaGraphT *graph) {
  MS_ASSERT(matMulPath != nullptr);
  MS_ASSERT(baPath != nullptr);
  MS_ASSERT(graph != nullptr);

  MS_ASSERT(graph->nodes.size() > matMulPath->nodeIdx);
  auto &matMulNode = graph->nodes.at(matMulPath->nodeIdx);
  MS_ASSERT(matMulNode != nullptr);
  auto baNode = graph->nodes.at(baPath->nodeIdx).get();
  MS_ASSERT(baNode != nullptr);

  // check biasTensor
  auto baWeightTensorIdxes = baNode->inputIndex;
  if (baWeightTensorIdxes.size() != BIASADD_OP_INPUT_NUM) {
    MS_LOG(ERROR) << "input number is invalid! node: " << baNode->name.c_str();
    return RET_ERROR;
  }
  MS_ASSERT(graph->allTensors.size() > baWeightTensorIdxes.at(BIASADD_OP_BIAS_INDEX));
  auto &biasTensor = graph->allTensors.at(baWeightTensorIdxes.at(BIASADD_OP_BIAS_INDEX));
  MS_ASSERT(biasTensor != nullptr);
  auto biasDims = biasTensor->dims;
  // if biasTensor is a scaler
  if (biasDims.empty() && biasTensor->data.data() == nullptr) {
    MS_LOG(ERROR) << "bias tensor is invalid, node: " << baNode->name.c_str();
    return RET_ERROR;
  }
  if (!biasDims.empty() && biasDims.size() != BIASADD_WEIGHT_SHAPE_SIZE) {
    MS_LOG(ERROR) << "BiasAdd bias tensor should has one dimension, current number of dimension " << biasDims.size()
                  << ". or bias tensor is a scaler";
    return RET_ERROR;
  }
  // add biasTensor to matmul
  matMulNode->inputIndex.emplace_back(baWeightTensorIdxes.at(BIASADD_OP_BIAS_INDEX));
  baNode->inputIndex.erase(baNode->inputIndex.begin() + BIASADD_OP_BIAS_INDEX);

  return RET_OK;
}

MatMulBiasAddFusionPass::~MatMulBiasAddFusionPass() = default;
}  // namespace lite
}  // namespace mindspore
