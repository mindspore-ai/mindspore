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

#include "tools/converter/legacy_optimizer/fusion/conv_activation_fusion_pass.h"
#include <memory>
#include <string>
#include <unordered_map>
#include "utils/log_adapter.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "tools/common/graph_util.h"
#include "src/common/op_utils.h"

namespace mindspore {
namespace lite {
#define CONV_ACTIVATION_MATCH_PATH_LEN 2

STATUS ConvActivationFusionPass::DefinePattern() {
  auto convOp = std::make_shared<PatternOp>();
  convOp->id = kConvName;
  convOp->types = {schema::PrimitiveType_Conv2D, schema::PrimitiveType_DepthwiseConv2D};
  auto actOp = std::make_shared<PatternOp>();
  actOp->id = ACTIVATION_NAME;
  actOp->types = {schema::PrimitiveType_Activation};
  actOp->left = convOp;

  std::unique_ptr<FusionPattern> fusionPattern(new (std::nothrow) FusionPattern("ConvActivationFusion"));
  if (fusionPattern == nullptr) {
    MS_LOG(ERROR) << "new fusionPattern failed";
    return RET_ERROR;
  }
  fusionPattern->AddPatternOp(convOp);
  fusionPattern->AddPatternOp(actOp);
  fusionPattern->Finish();

  this->patterns.emplace_back(fusionPattern.release());

  return RET_OK;
}

// 1. change attr of conv
// 2. delete Activation node
STATUS ConvActivationFusionPass::DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                                          std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  if (matchedPath.size() != CONV_ACTIVATION_MATCH_PATH_LEN) {
    MS_LOG(ERROR) << "Conv-Activation-Fusion should have two NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }

  auto convPath = matchedPath[kConvName];
  auto actPath = matchedPath[ACTIVATION_NAME];
  auto &convNode = graph->nodes.at(convPath->nodeIdx);
  auto &actNode = graph->nodes.at(actPath->nodeIdx);

  // todo if combine conv_relu_fusion and conv_relu6_fusion to conv_activation_fusion
  if (actNode->primitive->value.AsActivation()->type != this->activationType) {
    return RET_NO_CHANGE;
  }

  if (convNode->primitive->value.type == schema::PrimitiveType_Conv2D) {
    convNode->primitive->value.AsConv2D()->activationType = this->activationType;
  } else if (convNode->primitive->value.type == schema::PrimitiveType_DepthwiseConv2D) {
    convNode->primitive->value.AsDepthwiseConv2D()->activationType = this->activationType;
  } else {
    MS_LOG(ERROR) << "Unsupported opType, " << convNode->primitive->value.type;
    return RET_ERROR;
  }

  // remove activation node
  MergeNodeAttrFromPost(convNode, actNode);
  auto status = IsolateOneWayNode(graph, actPath->nodeIdx);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode failed, subGraph: " << actPath->subGraphIdx << ", node: " << actPath->nodeIdx
                  << ", error: " << status;
    return status;
  }

  return RET_OK;
}

STATUS ConvActivationFusionPass::Run(schema::MetaGraphT *graph) {
  SetActivationType();
  return FusionPass::Run(graph);
}

}  // namespace lite
}  // namespace mindspore
