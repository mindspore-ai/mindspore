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

#include "tools/converter/parser/onnx/onnx_reduce_parser.h"
#include <memory>
#include <vector>
#include "ops/fusion/reduce_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxReduceParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::ReduceFusion>();

  prim->set_keep_dims(true);

  std::vector<int32_t> axes = {};
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "axes") {
      const int &size = onnx_node_attr.ints_size();
      for (int i = 0; i < size; ++i) {
        axes.push_back(onnx_node_attr.ints(i));
      }
    } else if (attribute_name == "keepdims") {
      prim->set_keep_dims(static_cast<bool>(onnx_node_attr.i()));
    }
  }
  prim->AddAttr("axes", MakeValue(axes));

  const auto &type = onnx_node.op_type();
  if (type == "ReduceMean") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Mean);
  } else if (type == "ReduceMax") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Max);
  } else if (type == "ReduceMin") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Min);
  } else if (type == "ReduceSum") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Sum);
  } else if (type == "ReduceProd") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Prod);
  } else if (type == "ReduceSumSquare") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Sum_Square);
  } else {
    MS_LOG(ERROR) << "unsupported reduce type: " << type;
    return nullptr;
  }

  return prim.release();
}

OnnxNodeRegistrar g_onnxReduceMeanParser("ReduceMean", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceMaxParser("ReduceMax", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceMinParser("ReduceMin", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceProdParser("ReduceProd", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceSumParser("ReduceSum", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceSumSquareParser("ReduceSumSquare", new OnnxReduceParser());
}  // namespace lite
}  // namespace mindspore
