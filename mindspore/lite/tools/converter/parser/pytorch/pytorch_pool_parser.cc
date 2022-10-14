/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/pytorch/pytorch_pool_parser.h"
#include <memory>
#include <vector>
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fusion/max_pool_fusion.h"
#include "include/registry/converter_context.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
int SetAttrsForPool(const torch::jit::Node *torch_node, PrimitiveCPtr prim_c) {
  MS_ASSERT(torch_node != nullptr && prim_c != nullptr);
  auto node_type = PytorchNodeParser::GetTorchNodeType(torch_node);
  prim_c->AddAttr(ops::kPadMode, MakeValue(static_cast<int64_t>(mindspore::PadMode::PAD)));
  auto kernels = PytorchNodeParser::GetValueFromConstNode<std::vector<int64_t>>(torch_node->input(1));
  prim_c->AddAttr(ops::kKernelSize, MakeValue(kernels));
  if (node_type.find("adaptive") != node_type.npos) {
    std::vector<int64_t> out_kernels = {1, 1};
    if (out_kernels == kernels) {
      prim_c->AddAttr(ops::kGlobal, MakeValue(true));
    } else {
      MS_LOG(ERROR) << "Unsupported adaptive average pool with output kernels: " << kernels;
      return RET_ERROR;
    }
  }

  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  if (torch_node->inputs().size() > THIRD_INPUT) {
    strides = PytorchNodeParser::GetValueFromConstNode<std::vector<int64_t>>(torch_node->input(THIRD_INPUT));
  }
  if (torch_node->inputs().size() > FOURTH_INPUT) {
    pads = PytorchNodeParser::GetValueFromConstNode<std::vector<int64_t>>(torch_node->input(FOURTH_INPUT));
    if (pads.size() == DIMENSION_2D) {
      pads.push_back(pads.at(1));
      pads.insert(pads.begin(), pads.at(0));
    }
  }
  if (strides.empty()) {
    strides = {1, 1};
  }
  prim_c->AddAttr(ops::kStrides, MakeValue(strides));
  if (pads.empty()) {
    pads = {0, 0, 0, 0};
  }
  prim_c->AddAttr(ops::kPad, MakeValue(pads));

  mindspore::RoundMode round_mode = mindspore::RoundMode::FLOOR;
  if (torch_node->inputs().size() > SIXTH_INPUT) {
    round_mode = PytorchNodeParser::GetValueFromConstNode<bool>(torch_node->input(SIXTH_INPUT))
                   ? mindspore::RoundMode::CEIL
                   : round_mode;
  }
  prim_c->AddAttr(ops::kRoundMode, MakeValue(static_cast<int64_t>(round_mode)));
  prim_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(mindspore::Format::NCHW));
  prim_c->AddAttr(ops::kFmkType, MakeValue(static_cast<int>(converter::FmkType::kFmkTypePytorch)));
  return RET_OK;
}
PrimitiveCPtr PytorchAvgPoolParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::AvgPoolFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  input_indices->push_back(0);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  if (SetAttrsForPool(torch_node, prim_c) != lite::RET_OK) {
    MS_LOG(ERROR) << "Set attributes for pooling failed.";
    return nullptr;
  }
  return prim_c;
}

PrimitiveCPtr PytorchMaxPoolParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::MaxPoolFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  input_indices->push_back(0);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  if (SetAttrsForPool(torch_node, prim_c) != lite::RET_OK) {
    MS_LOG(ERROR) << "Set attributes for pooling failed.";
    return nullptr;
  }
  return prim_c;
}

PytorchNodeRegistrar g_pytorchAvgPoolParser("avg_pool2d", new PytorchAvgPoolParser());
PytorchNodeRegistrar g_pytorchAdaptiveAvgPoolParser("adaptive_avg_pool2d", new PytorchAvgPoolParser());
PytorchNodeRegistrar g_pytorchMaxPoolParser("max_pool2d", new PytorchMaxPoolParser());
PytorchNodeRegistrar g_pytorchAdaptiveMaxPoolParser("adaptive_max_pool2d", new PytorchMaxPoolParser());
}  // namespace lite
}  // namespace mindspore
