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
#include <memory>
#include "tools/converter/legacy_optimizer/fusion/format_trans_fusion_pass.h"
#include "src/common/log_adapter.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace {
std::vector<int> nchw2nhwc_perm = {0, 2, 3, 1};
std::vector<int> nhwc2nchw_perm = {0, 3, 1, 2};
}  // namespace
namespace lite {
#define kFormatTransMatchPathLen2 2
#define kFormatTransMatchPathLen3 3

STATUS FormatTransFusionPass::DefinePattern() {
  // nchw2nhwc + nhwc2nchw  ||  nhwc2nchw + nchw2nhwc
  {
    auto transpose1 = std::make_shared<PatternOp>();
    transpose1->id = kFormatTransTranspose1;
    transpose1->types = {PrimitiveType_Transpose};
    auto transpose2 = std::make_shared<PatternOp>();
    transpose2->id = kFormatTransTranspose2;
    transpose2->types = {PrimitiveType_Transpose};

    transpose2->left = transpose1;
    auto pattern = std::make_unique<FusionPattern>(kNc2NhAndNh2NcFusionPattern);
    if (pattern == nullptr) {
      MS_LOG(ERROR) << "new " << kNc2NhAndNh2NcFusionPattern << "failed";
      return RET_ERROR;
    }
    pattern->AddPatternOp(transpose1);
    pattern->AddPatternOp(transpose2);
    pattern->Finish();
    this->patterns.emplace_back(pattern.release());
  }
  // nhwc2nchw + QuantDtypeCast + nchw2nhwc  ||  nchw2nhwc + QuantDtypeCast + nhwc2nchw
  {
    auto transpose1 = std::make_shared<PatternOp>();
    transpose1->id = kFormatTransTranspose1;
    transpose1->types = {PrimitiveType_Transpose};
    auto passOp = std::make_shared<PatternOp>();
    passOp->id = kFormatTransPassOp;
    passOp->types = {PrimitiveType_QuantDTypeCast};
    auto transpose2 = std::make_shared<PatternOp>();
    transpose2->id = kFormatTransTranspose2;
    transpose2->types = {PrimitiveType_Transpose};

    passOp->left = transpose2;
    transpose1->left = passOp;
    auto pattern = std::make_unique<FusionPattern>(kNh2NcAndNc2NhPassFusionPattern);
    if (pattern == nullptr) {
      MS_LOG(ERROR) << "new " << kNh2NcAndNc2NhPassFusionPattern << " failed";
      return RET_ERROR;
    }
    pattern->AddPatternOp(transpose1);
    pattern->AddPatternOp(passOp);
    pattern->AddPatternOp(transpose2);
    pattern->Finish();
    this->patterns.emplace_back(pattern.release());
  }
  return RET_OK;
}

STATUS FormatTransFusionPass::Run(schema::MetaGraphT *graph) { return FusionPass::Run(graph); }

STATUS FormatTransFusionPass::DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                                       std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  if (matchedPath.size() != kFormatTransMatchPathLen2 && matchedPath.size() != kFormatTransMatchPathLen3) {
    MS_LOG(ERROR) << "schema::Format-Transform-Fusion should have " << kFormatTransMatchPathLen2 << " or "
                  << kFormatTransMatchPathLen3 << " NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }

  std::shared_ptr<Path> srcPath = matchedPath[kFormatTransTranspose1];
  std::shared_ptr<Path> dstPath = matchedPath[kFormatTransTranspose2];
  if (srcPath == nullptr || dstPath == nullptr) {
    MS_LOG(ERROR) << "srcPath or dstPath is failed to get";
    return RET_ERROR;
  }
  auto &srcNode = graph->nodes.at(srcPath->nodeIdx);
  auto &dstNode = graph->nodes.at(dstPath->nodeIdx);
  MS_ASSERT(srcNode != nullptr);
  MS_ASSERT(dstNode != nullptr);
  auto src_perm = GetTransposePerm(graph, srcNode);
  auto dst_perm = GetTransposePerm(graph, dstNode);
  bool isNc2NhAndNh2Nc = src_perm == nchw2nhwc_perm && dst_perm == nhwc2nchw_perm;
  bool isNh2NcAndNc2Nh = src_perm == nhwc2nchw_perm && dst_perm == nchw2nhwc_perm;
  if (isNc2NhAndNh2Nc || isNh2NcAndNc2Nh) {
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
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
