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

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "tools/converter/legacy_optimizer/fusion/format_trans_permute_fusion_pass.h"
#include "utils/log_adapter.h"
#include "securec/include/securec.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "mindspore/lite/schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
#define kFormatTransPermuteMatchPathLen 2

STATUS FormatTransPermuteFusionPass::DefinePattern() {
  // format trans + permute
  {
    auto formatTransOp = std::make_shared<PatternOp>();
    formatTransOp->id = kFormatTransformOp;
    formatTransOp->types = {PrimitiveType_Nchw2Nhwc, PrimitiveType_Nhwc2Nchw};
    auto permuteOp = std::make_shared<PatternOp>();
    permuteOp->id = kPermuteOp;
    permuteOp->types = {PrimitiveType_Permute};

    permuteOp->left = formatTransOp;
    std::unique_ptr<FusionPattern> formatTransPermuteFusionPattern(new (std::nothrow)
                                                                     FusionPattern(kFormatTrans2PermuteFusionPattern));
    if (formatTransPermuteFusionPattern == nullptr) {
      MS_LOG(ERROR) << "new " << kFormatTrans2PermuteFusionPattern << " failed";
      return RET_ERROR;
    }
    formatTransPermuteFusionPattern->AddPatternOp(formatTransOp);
    formatTransPermuteFusionPattern->AddPatternOp(permuteOp);
    formatTransPermuteFusionPattern->Finish();
    this->patterns.emplace_back(formatTransPermuteFusionPattern.release());
  }
  // permute + format trans
  {
    auto formatTransOp = std::make_shared<PatternOp>();
    formatTransOp->id = kFormatTransformOp;
    formatTransOp->types = {PrimitiveType_Nchw2Nhwc, PrimitiveType_Nhwc2Nchw};
    auto permuteOp = std::make_shared<PatternOp>();
    permuteOp->id = kPermuteOp;
    permuteOp->types = {PrimitiveType_Permute};

    formatTransOp->left = permuteOp;
    std::unique_ptr<FusionPattern> permuteFormatTransFusionPattern(new (std::nothrow)
                                                                     FusionPattern(kPermute2FormatTransFusionPattern));
    if (permuteFormatTransFusionPattern == nullptr) {
      MS_LOG(ERROR) << "new " << kPermute2FormatTransFusionPattern << " failed";
      return RET_ERROR;
    }
    permuteFormatTransFusionPattern->AddPatternOp(formatTransOp);
    permuteFormatTransFusionPattern->AddPatternOp(permuteOp);
    permuteFormatTransFusionPattern->Finish();
    this->patterns.emplace_back(permuteFormatTransFusionPattern.release());
  }
  return RET_OK;
}

STATUS FormatTransPermuteFusionPass::Run(schema::MetaGraphT *graph) { return FusionPass::Run(graph); }

STATUS FormatTransPermuteFusionPass::DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                                              std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  if (matchedPath.size() != kFormatTransPermuteMatchPathLen) {
    MS_LOG(ERROR) << "Format-Transform-Permute-Fusion should have " << kFormatTransPermuteMatchPathLen
                  << " NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }

  std::shared_ptr<Path> formatTransPath = matchedPath[kFormatTransformOp];
  std::shared_ptr<Path> permutePath = matchedPath[kPermuteOp];
  if (formatTransPath == nullptr) {
    MS_LOG(ERROR) << "formatTransPath is failed to get";
    return RET_ERROR;
  }
  if (permutePath == nullptr) {
    MS_LOG(ERROR) << "permutePath is failed to get";
    return RET_ERROR;
  }
  auto &formatTransNode = graph->nodes.at(formatTransPath->nodeIdx);
  auto &permuteNode = graph->nodes.at(permutePath->nodeIdx);
  MS_ASSERT(formatTransNode != nullptr);
  MS_ASSERT(permuteNode != nullptr);
  auto formatTransType = formatTransNode->primitive->value.type;
  if (formatTransType != PrimitiveType_Nhwc2Nchw && formatTransType != PrimitiveType_Nchw2Nhwc) {
    MS_LOG(ERROR) << "FormatTransNode should be " << EnumNamePrimitiveType(PrimitiveType_Nhwc2Nchw) << " or "
                  << EnumNamePrimitiveType(PrimitiveType_Nchw2Nhwc) << ", but got "
                  << EnumNamePrimitiveType(formatTransType);
    return RET_ERROR;
  }
  MS_ASSERT(permuteNode->primitive != nullptr);
  auto permPrimitive = permuteNode->primitive->value.AsPermute();
  MS_ASSERT(permPrimitive != nullptr);
  auto perm = permPrimitive->order;
  if (perm.size() != 4) {
    return RET_OK;
  }
  std::vector<int64_t> nchw2nhwcPerm = {0, 2, 3, 1};
  std::vector<int64_t> nhwc2nchwPerm = {0, 3, 1, 2};
  if ((perm == nchw2nhwcPerm && formatTransType == PrimitiveType_Nhwc2Nchw) ||
      (perm == nhwc2nchwPerm && formatTransType == PrimitiveType_Nchw2Nhwc)) {
    auto status = IsolateOneWayNode(graph, formatTransPath->nodeIdx);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "IsolateOneWayNode failed, node: " << formatTransNode->name << ", error: " << status;
      return status;
    }

    status = IsolateOneWayNode(graph, permutePath->nodeIdx);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "IsolateOneWayNode failed, node: " << permuteNode->name << ", error: " << status;
      return status;
    }
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
