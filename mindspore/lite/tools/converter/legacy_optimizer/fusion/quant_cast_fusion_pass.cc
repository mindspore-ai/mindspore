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
#include <memory>
#include "tools/converter/legacy_optimizer/fusion/quant_cast_fusion_pass.h"
#include "src/common/log_adapter.h"
#include "securec/include/securec.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
#define kQuantCastMatchPathLen2 2
#define kQuantCastMatchPathLen3 3

STATUS QuantCastFusionPass::Run(MetaGraphT *graph) { return FusionPass::Run(graph); }

STATUS QuantCastFusionPass::DoFusion(MetaGraphT *graph, const std::string &patternName,
                                     std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  if (matchedPath.size() != kQuantCastMatchPathLen2 && matchedPath.size() != kQuantCastMatchPathLen3) {
    MS_LOG(ERROR) << "QuantDtypeCastFusion should have " << kQuantCastMatchPathLen2 << " or " << kQuantCastMatchPathLen3
                  << " NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }

  auto srcPath = matchedPath[kQuantCastSrcOp];
  MS_ASSERT(srcPath != nullptr);
  auto dstPath = matchedPath[kQuantCastDstOp];
  MS_ASSERT(dstPath != nullptr);
  auto srcNode = graph->nodes.at(srcPath->nodeIdx).get();
  MS_ASSERT(srcNode != nullptr);
  auto dstNode = graph->nodes.at(dstPath->nodeIdx).get();
  MS_ASSERT(dstNode != nullptr);

  if (srcNode->inputIndex.empty() && srcNode->outputIndex.empty()) {
    MS_LOG(DEBUG) << "srcNode " << srcNode->name.c_str() << " has been removed";
    return RET_NO_CHANGE;
  }
  if (dstNode->inputIndex.empty() && dstNode->outputIndex.empty()) {
    MS_LOG(DEBUG) << "dstNode " << dstNode->name.c_str() << " has been removed";
    return RET_NO_CHANGE;
  }

  auto srcAttr = srcNode->primitive->value.AsQuantDTypeCast();
  auto dstAttr = dstNode->primitive->value.AsQuantDTypeCast();
  MS_ASSERT(srcAttr != nullptr);
  MS_ASSERT(dstAttr != nullptr);
  if (srcAttr->dst_t != dstAttr->src_t) {
    MS_LOG(ERROR) << "srcNode and dstNode can not been fused";
    return RET_ERROR;
  }

  auto status = IsolateOneWayNode(graph, srcPath->nodeIdx);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode failed, node: " << srcNode->name.c_str() << ", error: " << status;
    return status;
  }

  if (srcAttr->src_t == dstAttr->dst_t) {
    status = IsolateOneWayNode(graph, dstPath->nodeIdx);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "IsolateOneWayNode failed, node: " << dstNode->name.c_str() << ", error: " << status;
      return status;
    }
  } else {
    dstAttr->src_t = srcAttr->src_t;
  }

  return RET_OK;
}

STATUS QuantCastFusionPass::DefinePattern() {
  // quantCast + quantCast
  {
    auto srcOp = std::make_shared<PatternOp>();
    srcOp->id = kQuantCastSrcOp;
    srcOp->types = {schema::PrimitiveType_QuantDTypeCast};
    auto dstOp = std::make_shared<PatternOp>();
    dstOp->id = kQuantCastDstOp;
    dstOp->types = {schema::PrimitiveType_QuantDTypeCast};
    dstOp->left = srcOp;

    std::unique_ptr<FusionPattern> fusionPattern(new (std::nothrow) FusionPattern(kQuantCastFusionPattern));
    if (fusionPattern == nullptr) {
      MS_LOG(ERROR) << "new fusionPattern failde";
      return RET_ERROR;
    }
    fusionPattern->AddPatternOp(srcOp);
    fusionPattern->AddPatternOp(dstOp);
    fusionPattern->Finish();

    this->patterns.emplace_back(fusionPattern.release());
  }
  // quantCast + formatTrans + quantCast
  {
    auto srcOp = std::make_shared<PatternOp>();
    srcOp->id = kQuantCastSrcOp;
    srcOp->types = {schema::PrimitiveType_QuantDTypeCast};
    auto formatOp = std::make_shared<PatternOp>();
    formatOp->id = kFormatTransOp;
    formatOp->types = {PrimitiveType_Transpose};
    formatOp->left = srcOp;
    auto dstOp = std::make_shared<PatternOp>();
    dstOp->id = kQuantCastDstOp;
    dstOp->types = {schema::PrimitiveType_QuantDTypeCast};
    dstOp->left = formatOp;

    std::unique_ptr<FusionPattern> fusionPattern(new (std::nothrow) FusionPattern(kQuantCastPassFusionPattern));
    if (fusionPattern == nullptr) {
      MS_LOG(ERROR) << "new fusionPattern failde";
      return RET_ERROR;
    }
    fusionPattern->AddPatternOp(srcOp);
    fusionPattern->AddPatternOp(formatOp);
    fusionPattern->AddPatternOp(dstOp);
    fusionPattern->Finish();

    this->patterns.emplace_back(fusionPattern.release());
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
