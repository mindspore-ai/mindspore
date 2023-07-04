/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_roialign_parser.h"
#include <memory>
#include "ops/roi_align.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kInputSize3 = 3;
}  // namespace

PrimitiveCPtr OnnxRoiAlignParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::ROIAlign>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  if (onnx_node.input_size() != kInputSize3) {
    MS_LOG(ERROR) << "The input size of ROIAlign must be 3";
    return nullptr;
  }
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "output_height") {
      prim->set_pooled_height(static_cast<int64_t>(onnx_node_attr.i()));
    } else if (attribute_name == "output_width") {
      prim->set_pooled_width(static_cast<int64_t>(onnx_node_attr.i()));
    } else if (attribute_name == "spatial_scale") {
      prim->set_spatial_scale(onnx_node_attr.f());
    } else if (attribute_name == "sampling_ratio") {
      prim->set_sample_num(static_cast<int64_t>(onnx_node_attr.i()));
    }
    prim->set_roi_end_mode(static_cast<int64_t>(0));
  }
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxRoiAlignParser("RoiAlign", new OnnxRoiAlignParser());
}  // namespace lite
}  // namespace mindspore
