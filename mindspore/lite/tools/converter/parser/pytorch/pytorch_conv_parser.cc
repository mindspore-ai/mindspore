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

#include "tools/converter/parser/pytorch/pytorch_conv_parser.h"
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include "ops/fusion/conv2d_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
PrimitiveCPtr PytorchConvParser::Parse(const torch::jit::Node *torch_node, std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<ops::Conv2DFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto bias = torch_node->input(kBiasIndex);
  MS_CHECK_TRUE_RET(bias != nullptr, nullptr);
  // the bias is noneType
  size_t input_size = bias->isCompleteTensor() ? kInputSize2 : kInputSize1;
  input_indices->resize(input_size);
  std::iota(input_indices->begin(), input_indices->end(), 0);

  auto stride = PytorchNodeParser::GetValueFromConstNode<std::vector<int64_t>>(torch_node->input(FOURTH_INPUT));
  prim->set_stride(stride);
  auto dilation = PytorchNodeParser::GetValueFromConstNode<std::vector<int64_t>>(torch_node->input(SIXTH_INPUT));
  prim->set_dilation(dilation);

  auto padding = PytorchNodeParser::GetValueFromConstNode<std::vector<int64_t>>(torch_node->input(FIFTH_INPUT));
  if (padding.size() == DIMENSION_2D) {
    padding.push_back(padding.at(1));
    padding.insert(padding.begin(), padding.at(0));
  }
  prim->set_pad_list(padding);

  auto group = PytorchNodeParser::GetValueFromConstNode<int64_t>(torch_node->input(8));
  prim->set_group(group);

  prim->set_pad({0, 0, 0, 0});
  mindspore::PadMode pad_mode = mindspore::PadMode::PAD;
  prim->set_pad_mode(pad_mode);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  mindspore::Format format = mindspore::Format::NCHW;
  prim_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(format));
  bool conv1d = stride.size() == 1;
  if (conv1d) {
    prim_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(NCW));
  }

  // parse activationType
  prim->set_activation_type(mindspore::ActivationType::NO_ACTIVATION);

  return prim->GetPrim();
}

PytorchNodeRegistrar g_pytorchConvParser("conv2d", new PytorchConvParser());
PytorchNodeRegistrar g_pytorchConvolutionParser("convolution", new PytorchConvParser());
}  // namespace mindspore::lite
