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

#include "tools/converter/parser/onnx/onnx_forward_rasterize_parser.h"
#include <memory>
#include <map>
#include <utility>
#include <string>
#include <vector>
#include "ops/custom.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxForwardRasterizeParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Custom>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  std::vector<std::string> input_name = {"face_vertices"};
  std::vector<std::string> output_name = {"output_buffer"};
  prim->AddAttr("input_names", api::MakeValue(input_name));
  prim->AddAttr("output_names", api::MakeValue(output_name));
  prim->set_type("ForwardRasterize");
  prim->AddAttr("reg_op_name", api::MakeValue("ForwardRasterize"));

  int height = 256;
  int width = 256;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "h") {
      height = onnx_node_attr.i();
      prim->AddAttr("h", api::MakeValue(height));
    } else if (attribute_name == "w") {
      width = onnx_node_attr.i();
      prim->AddAttr("w", api::MakeValue(width));
    }
  }
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  // add "h" attr
  auto custom_h_str = std::to_string(height);
  std::vector<uint8_t> custom_h_vec(custom_h_str.begin(), custom_h_str.end());
  (void)custom_attrs.insert(std::make_pair("h", custom_h_vec));

  // add "w" attr
  auto custom_w_str = std::to_string(width);
  std::vector<uint8_t> custom_w_vec(custom_w_str.begin(), custom_w_str.end());
  (void)custom_attrs.insert(std::make_pair("w", custom_w_vec));

  prim->set_attr(custom_attrs);

  std::string attr_val_str(custom_h_vec.begin(), custom_h_vec.end());
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxForwardRasterizeParser("ForwardRasterize", new OnnxForwardRasterizeParser());
}  // namespace lite
}  // namespace mindspore
