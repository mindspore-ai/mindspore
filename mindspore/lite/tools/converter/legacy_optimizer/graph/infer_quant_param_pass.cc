/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "tools/converter/legacy_optimizer/graph/infer_quant_param_pass.h"
#include <vector>
#include <memory>
#include "src/common/utils.h"
#include "tools/converter/quantizer/calc_quant_param.h"
#include "tools/common/node_util.h"

namespace mindspore::lite {
STATUS InferQuantParamPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  auto *quantParamRegister = QuantParamCalcRegister::GetInstance();

  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    auto &node = *iter;
    MS_ASSERT(node != nullptr);
    if (node->quantType == schema::QuantType_WeightQuant) {
      continue;
    }
    DetermineNodeQuantType(*graph, node.get());
    if (node->quantType == schema::QuantType_AwareTraining) {
      continue;
    }
    if (GetCNodeTType(*node) == schema::PrimitiveType_FakeQuantWithMinMaxVars) {
      MS_ASSERT(false);
    }
    auto quantParamCalcer = quantParamRegister->GetQuantParamCalcer(GetCNodeTType(*node));
    if (quantParamCalcer == nullptr) {
      MS_LOG(DEBUG) << "Can not find QuantParamCalcer for " << node->name.c_str()
                    << ", type: " << GetCNodeTTypeName(*node).c_str() << " set node to QuantNone and skip";
      node->quantType = schema::QuantType_QUANT_NONE;
    } else {
      auto status = quantParamCalcer->Calc(graph, *node);
      if (status != RET_OK) {
        MS_LOG(DEBUG) << "quantParamCalcer failed: " << status << " node: " << node->name.c_str();
        node->quantType = schema::QuantType_QUANT_NONE;
      } else {
        node->quantType = schema::QuantType_AwareTraining;
      }
    }
  }
  return RET_OK;
}

void InferQuantParamPass::DetermineNodeQuantType(const schema::MetaGraphT &graph, schema::CNodeT *cnode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(cnode != nullptr);
  bool canQuant = true;
  for (auto &inputTensorIdx : cnode->inputIndex) {
    MS_ASSERT(graph.allTensors.size() > inputTensorIdx);
    auto &inTensor = graph.allTensors.at(inputTensorIdx);
    MS_ASSERT(inTensor != nullptr);
    if (inTensor->quantParams.empty() || inTensor->quantParams.front() == nullptr ||
        !inTensor->quantParams.front()->inited) {
      canQuant = false;
      break;
    }
  }

  for (auto &outTensorIdx : cnode->outputIndex) {
    MS_ASSERT(graph.allTensors.size() > outTensorIdx);
    auto &outTensor = graph.allTensors.at(outTensorIdx);
    MS_ASSERT(outTensor != nullptr);
    if (outTensor->quantParams.empty() || outTensor->quantParams.front() == nullptr ||
        !outTensor->quantParams.front()->inited) {
      canQuant = false;
      break;
    }
  }

  if (canQuant) {
    cnode->quantType = schema::QuantType_AwareTraining;
  } else {
    cnode->quantType = schema::QuantType_QUANT_NONE;
  }
}
}  // namespace mindspore::lite
