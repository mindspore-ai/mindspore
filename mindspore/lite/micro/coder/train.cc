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

#include "coder/train.h"
#include <memory>
#include <set>
#include <array>
#include <queue>
#include <string>
#include <vector>
#include <algorithm>
#include "schema/ops_generated.h"
#include "src/common/prim_util.h"

namespace mindspore::lite::micro {

std::set<OperatorCoder *> FindInferenceOpcoders(OperatorCoder *edge) {
  std::set<OperatorCoder *> subgraph;
  std::queue<OperatorCoder *> to_visit;
  to_visit.push(edge);
  while (!to_visit.empty()) {
    size_t size = to_visit.size();
    for (size_t i = 0; i < size; ++i) {
      OperatorCoder *curr = to_visit.front();
      to_visit.pop();
      if (subgraph.find(curr) != subgraph.end()) {
        continue;
      }
      subgraph.insert(curr);
      for (const auto &op : curr->input_ops()) {
        to_visit.push(op);
      }
    }
  }
  auto item = subgraph.find(edge);
  if (item == subgraph.end()) {
    MS_LOG(ERROR) << "failed to find the edge in the subgraph";
    return subgraph;
  }
  // erase edge operator coder from subgraph
  subgraph.erase(item);
  return subgraph;
}

int Train::TransformGraphForTrain(CoderContext *context, const std::vector<std::unique_ptr<OperatorCoder>> &op_coders) {
  const std::array<int, 6> loss_types = {schema::PrimitiveType_SparseSoftmaxCrossEntropyWithLogits,
                                         schema::PrimitiveType_BinaryCrossEntropy,
                                         schema::PrimitiveType_SmoothL1Loss,
                                         schema::PrimitiveType_SmoothL1LossGrad,
                                         schema::PrimitiveType_SigmoidCrossEntropyWithLogits,
                                         schema::PrimitiveType_SigmoidCrossEntropyWithLogitsGrad};
  OperatorCoder *loss_op = nullptr;
  for (const auto &opcoder : op_coders) {
    const Model::Node *node = opcoder->node();
    int primitive_type = GetPrimitiveType(node->primitive_);
    auto item = std::find(loss_types.begin(), loss_types.end(), primitive_type);
    if (item != loss_types.end()) {
      loss_op = opcoder.get();
      break;
    }
  }
  MS_CHECK_PTR(loss_op);
  size_t op_num = op_coders.size();
  std::vector<std::string> code_blocks = context->code_blocks();
  if (op_num != code_blocks.size()) {
    MS_LOG(INFO) << "the number of code blocks and op coders is not equal";
    return RET_ERROR;
  }
  std::set<OperatorCoder *> inference_ops = FindInferenceOpcoders(loss_op);
  std::vector<std::string> inferences_blocks;
  std::vector<std::string> train_blocks;
  for (size_t i = 0; i < op_num; ++i) {
    auto &opcoder = op_coders.at(i);
    std::string block = code_blocks.at(i);
    if (inference_ops.find(opcoder.get()) != inference_ops.end()) {
      inferences_blocks.push_back(block);
    }
    train_blocks.push_back(block);
  }
  context->set_inference_blocks(inferences_blocks);
  context->set_train_blocks(train_blocks);
  return RET_OK;
}

}  // namespace mindspore::lite::micro
