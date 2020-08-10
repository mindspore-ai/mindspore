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
#include <unordered_map>
#include <memory>
#include "tools/converter/legacy_optimizer/fusion/format_trans_fusion_pass.h"
#include "utils/log_adapter.h"
#include "securec/include/securec.h"
// #include "utils/log_adapter.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "mindspore/lite/schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
#define kFormatTransMatchPathLen2 2
#define kFormatTransMatchPathLen3 3

STATUS FormatTransFusionPass::DefinePattern() {
  // nchw2nhwc + nhwc2nchw
  {
    auto nc2nhOp = std::make_shared<PatternOp>();
    nc2nhOp->id = kFormatTransNc2NhOp;
    nc2nhOp->types = {PrimitiveType_Nchw2Nhwc};
    auto nh2ncOp = std::make_shared<PatternOp>();
    nh2ncOp->id = kFormatTransNh2NcOp;
    nh2ncOp->types = {PrimitiveType_Nhwc2Nchw};

    nh2ncOp->left = nc2nhOp;
    std::unique_ptr<FusionPattern> nc2NhAndNh2NcFusionPattern(new (std::nothrow)
                                                                FusionPattern(kNc2NhAndNh2NcFusionPattern));
    if (nc2NhAndNh2NcFusionPattern == nullptr) {
      // MS_LOG(ERROR) << "new %s failed", kNc2NhAndNh2NcFusionPattern);
      return RET_ERROR;
    }
    nc2NhAndNh2NcFusionPattern->AddPatternOp(nc2nhOp);
    nc2NhAndNh2NcFusionPattern->AddPatternOp(nh2ncOp);
    nc2NhAndNh2NcFusionPattern->Finish();
    this->patterns.emplace_back(nc2NhAndNh2NcFusionPattern.release());
  }
  // nchw2nhwc + QuantDtypeCast + nhwc2nchw
  {
    auto nc2nhOp = std::make_shared<PatternOp>();
    nc2nhOp->id = kFormatTransNc2NhOp;
    nc2nhOp->types = {PrimitiveType_Nchw2Nhwc};
    auto passOp = std::make_shared<PatternOp>();
    passOp->id = kFormatTransPassOp;
    passOp->types = {PrimitiveType_QuantDTypeCast};
    auto nh2ncOp = std::make_shared<PatternOp>();
    nh2ncOp->id = kFormatTransNh2NcOp;
    nh2ncOp->types = {PrimitiveType_Nhwc2Nchw};

    passOp->left = nc2nhOp;
    nh2ncOp->left = passOp;
    std::unique_ptr<FusionPattern> nc2NhAndNh2NcPassFusionPattern(new FusionPattern(kNc2NhAndNh2NcPassFusionPattern));
    if (nc2NhAndNh2NcPassFusionPattern == nullptr) {
      // MS_LOG(ERROR) << "new %s failed", kNc2NhAndNh2NcPassFusionPattern);
      return RET_ERROR;
    }
    nc2NhAndNh2NcPassFusionPattern->AddPatternOp(nc2nhOp);
    nc2NhAndNh2NcPassFusionPattern->AddPatternOp(passOp);
    nc2NhAndNh2NcPassFusionPattern->AddPatternOp(nh2ncOp);
    nc2NhAndNh2NcPassFusionPattern->Finish();
    this->patterns.emplace_back(nc2NhAndNh2NcPassFusionPattern.release());
  }
  // nhwc2nchw + nchw2nhwc
  {
    auto nc2nhOp = std::make_shared<PatternOp>();
    nc2nhOp->id = kFormatTransNc2NhOp;
    nc2nhOp->types = {PrimitiveType_Nchw2Nhwc};
    auto nh2ncOp = std::make_shared<PatternOp>();
    nh2ncOp->id = kFormatTransNh2NcOp;
    nh2ncOp->types = {PrimitiveType_Nhwc2Nchw};

    nc2nhOp->left = nh2ncOp;
    std::unique_ptr<FusionPattern> nh2NcAndNc2NhFusionPattern(new (std::nothrow)
                                                                FusionPattern(kNh2NcAndNc2NhFusionPattern));
    if (nh2NcAndNc2NhFusionPattern == nullptr) {
      // MS_LOG(ERROR) << "new %s failed", kNh2NcAndNc2NhFusionPattern);
      return RET_ERROR;
    }
    nh2NcAndNc2NhFusionPattern->AddPatternOp(nh2ncOp);
    nh2NcAndNc2NhFusionPattern->AddPatternOp(nc2nhOp);
    nh2NcAndNc2NhFusionPattern->Finish();
    this->patterns.emplace_back(nh2NcAndNc2NhFusionPattern.release());
  }
  // nhwc2nchw + QuantDtypeCast + nchw2nhwc
  {
    auto nc2nhOp = std::make_shared<PatternOp>();
    nc2nhOp->id = kFormatTransNc2NhOp;
    nc2nhOp->types = {PrimitiveType_Nchw2Nhwc};
    auto passOp = std::make_shared<PatternOp>();
    passOp->id = kFormatTransPassOp;
    passOp->types = {PrimitiveType_QuantDTypeCast};
    auto nh2ncOp = std::make_shared<PatternOp>();
    nh2ncOp->id = kFormatTransNh2NcOp;
    nh2ncOp->types = {PrimitiveType_Nhwc2Nchw};

    passOp->left = nh2ncOp;
    nc2nhOp->left = passOp;
    std::unique_ptr<FusionPattern> nh2NcAndNc2NhPassFusionPattern(new (std::nothrow)
                                                                    FusionPattern(kNh2NcAndNc2NhPassFusionPattern));
    if (nh2NcAndNc2NhPassFusionPattern == nullptr) {
      MS_LOG(ERROR) << "new " << kNh2NcAndNc2NhPassFusionPattern << " failed";
      return RET_ERROR;
    }
    nh2NcAndNc2NhPassFusionPattern->AddPatternOp(nh2ncOp);
    nh2NcAndNc2NhPassFusionPattern->AddPatternOp(passOp);
    nh2NcAndNc2NhPassFusionPattern->AddPatternOp(nc2nhOp);
    nh2NcAndNc2NhPassFusionPattern->Finish();
    this->patterns.emplace_back(nh2NcAndNc2NhPassFusionPattern.release());
  }
  return RET_OK;
}

STATUS FormatTransFusionPass::Run(schema::MetaGraphT *graph) { return FusionPass::Run(graph); }

STATUS FormatTransFusionPass::DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                                       std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  if (matchedPath.size() != kFormatTransMatchPathLen2 && matchedPath.size() != kFormatTransMatchPathLen3) {
    MS_LOG(ERROR) << "Format-Transform-Fusion should have " << kFormatTransMatchPathLen2 << " or "
                  << kFormatTransMatchPathLen3 << " NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }

  std::shared_ptr<Path> srcPath;
  std::shared_ptr<Path> dstPath;
  if (patternName == kNc2NhAndNh2NcFusionPattern || patternName == kNc2NhAndNh2NcPassFusionPattern) {
    srcPath = matchedPath[kFormatTransNc2NhOp];
    dstPath = matchedPath[kFormatTransNh2NcOp];
  } else if (patternName == kNh2NcAndNc2NhFusionPattern || patternName == kNh2NcAndNc2NhPassFusionPattern) {
    srcPath = matchedPath[kFormatTransNh2NcOp];
    dstPath = matchedPath[kFormatTransNc2NhOp];
  } else {
    MS_ASSERT(false);
  }
  if (srcPath == nullptr) {
    MS_LOG(ERROR) << "srcPath is failed to get";
    return RET_ERROR;
  }
  if (dstPath == nullptr) {
    MS_LOG(ERROR) << "dstPath is failed to get";
    return RET_ERROR;
  }
  auto srcNode = graph->nodes.at(srcPath->nodeIdx).get();
  auto dstNode = graph->nodes.at(dstPath->nodeIdx).get();
  MS_ASSERT(srcNode != nullptr);
  MS_ASSERT(dstNode != nullptr);
  if (patternName == kNc2NhAndNh2NcFusionPattern || patternName == kNc2NhAndNh2NcPassFusionPattern) {
    MS_ASSERT(GetCNodeTType(*srcNode) == schema::PrimitiveType_Nchw2Nhwc);
    MS_ASSERT(GetCNodeTType(*dstNode) == schema::PrimitiveType_Nhwc2Nchw);
  } else if (patternName == kNh2NcAndNc2NhFusionPattern || patternName == kNh2NcAndNc2NhPassFusionPattern) {
    MS_ASSERT(GetCNodeTType(*srcNode) == schema::PrimitiveType_Nhwc2Nchw);
    MS_ASSERT(GetCNodeTType(*dstNode) == schema::PrimitiveType_Nchw2Nhwc);
  } else {
    MS_ASSERT(false);
  }

  auto status = IsolateOneWayNode(graph, srcPath->nodeIdx);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode failed, node: " << srcNode->name << ", error: " << status;
    return status;
  }

  status = IsolateOneWayNode(graph, dstPath->nodeIdx);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "IsolateOneWayNode failed, node: " << dstNode->name << ", error: " << status;
    return status;
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore


