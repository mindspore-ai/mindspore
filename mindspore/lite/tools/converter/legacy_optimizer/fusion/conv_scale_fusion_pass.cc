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
#include "tools/converter/legacy_optimizer/fusion/conv_scale_fusion_pass.h"
#include "securec/include/securec.h"
#include "utils/log_adapter.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
#define SCALE_OP_NO_BIAS_WEIGHT_NUM 1
#define SCALE_OP_HAS_BIAS_WEIGHT_NUM 2

#define SCALE_OP_SCALE_INDEX_IN_WEIGHT 0
#define SCALE_OP_BIAS_INDEX_IN_WEIGHT 1

STATUS ConvScaleFusionPass::DefinePattern() {
  auto convOp = std::make_shared<PatternOp>();
  convOp->id = kConvName;
  convOp->types = {schema::PrimitiveType_Conv2D, schema::PrimitiveType_DepthwiseConv2D};
  auto scaleOp = std::make_shared<PatternOp>();
  scaleOp->id = DST_NAME;
  scaleOp->types = {schema::PrimitiveType_Scale};
  scaleOp->left = convOp;

  std::unique_ptr<FusionPattern> fusionPattern(new (std::nothrow) FusionPattern("ConvScaleFusion"));
  if (fusionPattern == nullptr) {
    MS_LOG(ERROR) << "new fusionPattern failed";
    return RET_ERROR;
  }
  fusionPattern->AddPatternOp(convOp);
  fusionPattern->AddPatternOp(scaleOp);
  fusionPattern->Finish();

  this->patterns.emplace_back(fusionPattern.release());

  return RET_OK;
}

STATUS ConvScaleFusionPass::DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                                     std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  return ConvScaleBiasFusionPass::DoFusion(graph, patternName, matchedPath);
}

STATUS ConvScaleFusionPass::Run(schema::MetaGraphT *graph) { return ConvScaleBiasFusionPass::Run(graph); }

STATUS ConvScaleFusionPass::GetTransParam(schema::MetaGraphT *graph, std::shared_ptr<Path> scalePath,
    int32_t kernelNum) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(scalePath != nullptr);

  auto scaleNode = graph->nodes.at(scalePath->nodeIdx).get();
  MS_ASSERT(scaleNode != nullptr);
  auto scaleWeightTensorIdxes = scaleNode->inputIndex;
  scaleWeightTensorIdxes.erase(scaleWeightTensorIdxes.begin());

  schema::TensorT *scaleTensor = nullptr;
  schema::TensorT *biasTensor = nullptr;

  if (scaleWeightTensorIdxes.size() == SCALE_OP_NO_BIAS_WEIGHT_NUM) {
    scaleTensor = graph->allTensors.at(scaleWeightTensorIdxes[SCALE_OP_SCALE_INDEX_IN_WEIGHT]).get();
  } else if (scaleWeightTensorIdxes.size() == SCALE_OP_HAS_BIAS_WEIGHT_NUM) {
    scaleTensor = graph->allTensors.at(scaleWeightTensorIdxes[SCALE_OP_SCALE_INDEX_IN_WEIGHT]).get();
    biasTensor = graph->allTensors.at(scaleWeightTensorIdxes[SCALE_OP_BIAS_INDEX_IN_WEIGHT]).get();
  } else {
    MS_LOG(ERROR) << "Scale should has %d or %d weight tensors, current number of weight tensors %zu";
          //  SCALE_OP_NO_BIAS_WEIGHT_NUM, SCALE_OP_HAS_BIAS_WEIGHT_NUM, scaleWeightTensorIdxes.size());
    return RET_ERROR;
  }

  if (scaleTensor == nullptr) {
    MS_LOG(ERROR) << "Scale's scale tensor is nullptr";
    return RET_ERROR;
  }

  if (kernelNum != scaleTensor->data.size() * sizeof(uint8_t) / sizeof(float)) {
    MS_LOG(ERROR) << "conv kernel num %u is expected to be equal to scale size(%lu)";
    //, kernelNum, scaleTensor->data.size() * sizeof(uint8_t) / sizeof(float));
    return RET_ERROR;
  }

  const float *scaleData = reinterpret_cast<float *>(scaleTensor->data.data());

  if (0 != memcpy_s(transScale, kernelNum * sizeof(float), scaleData, kernelNum * sizeof(float))) {
    MS_LOG(ERROR) << "memcpy_s transScale failed";
    return RET_ERROR;
  }

  if (biasTensor != nullptr) {
    if (kernelNum != biasTensor->data.size() * sizeof(uint8_t) / sizeof(float)) {
      MS_LOG(ERROR) << "conv kernel num %u is expected to be equal to bias size(%lu)";
      //, kernelNum, biasTensor->data.size() * sizeof(uint8_t) / sizeof(float));
      return RET_ERROR;
    }

    const float *biasData = reinterpret_cast<float *>(biasTensor->data.data());

    if (0 != memcpy_s(transBias, kernelNum * sizeof(float), biasData, kernelNum * sizeof(float))) {
      MS_LOG(ERROR) << "memcpy_s transBias failed";
      return RET_ERROR;
    }
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore


