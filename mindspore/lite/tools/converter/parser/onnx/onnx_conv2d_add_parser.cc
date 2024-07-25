/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_conv2d_add_parser.h"
#include <string>
#include <memory>
#include <utility>
#include <vector>
#include "nnacl/op_base.h"
#include "ops/custom.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::lite {
PrimitiveCPtr OnnxConv2dAddParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Custom>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  std::vector<std::string> input_name = {"featureMap", "weight", "bias", "addTensor"};
  std::vector<std::string> output_name = {"addTensor"};
  prim->AddAttr(kAttrInputNames, api::MakeValue(input_name));
  prim->AddAttr(kAttrOutputNames, api::MakeValue(output_name));
  prim->set_type("Conv2dAdd");
  prim->AddAttr("reg_op_name", api::MakeValue("Conv2dAdd"));
  std::vector<int64_t> dilations;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == kAttrDilations) {
      if (onnx_node_attr.ints_size() == opt::kInputSizeTwo) {
        dilations.push_back(onnx_node_attr.ints(opt::kInputIndexZero));
        dilations.push_back(onnx_node_attr.ints(opt::kInputIndexOne));
      }
    }
    if (attribute_name == kAttrStrides) {
      if (onnx_node_attr.ints_size() == opt::kInputSizeTwo) {
        strides.push_back(onnx_node_attr.ints(opt::kInputIndexZero));
        strides.push_back(onnx_node_attr.ints(opt::kInputIndexOne));
      }
    }
    if (attribute_name == kAttrPads) {
      if (onnx_node_attr.ints_size() == opt::kInputSizeFour) {
        pads.push_back(onnx_node_attr.ints(opt::kInputIndexZero));
        pads.push_back(onnx_node_attr.ints(opt::kInputIndexTwo));
        pads.push_back(onnx_node_attr.ints(opt::kInputIndexOne));
        pads.push_back(onnx_node_attr.ints(opt::kInputIndexThree));
      }
    }
  }
  if (dilations.empty()) {
    MS_LOG(ERROR) << "Cannot get Attr dilations from ONNX!";
    return nullptr;
  }
  if (strides.empty()) {
    MS_LOG(ERROR) << "Cannot get Attr strides from ONNX!";
    return nullptr;
  }
  if (pads.empty()) {
    MS_LOG(ERROR) << "Cannot get Attr pads from ONNX!";
    return nullptr;
  }
  prim->AddAttr(kAttrStrides, api::MakeValue(strides));
  prim->AddAttr(kAttrPads, api::MakeValue(pads));
  prim->AddAttr(kAttrDilations, api::MakeValue(dilations));
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxConv2dAddParser("Conv2dAdd", new OnnxConv2dAddParser());
}  // namespace mindspore::lite
