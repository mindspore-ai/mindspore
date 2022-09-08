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
#include "tools/converter/parser/onnx/onnx_random_normal_parser.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "ops/random_normal.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxRandomNormalParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::RandomNormal>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "dtype") {
      auto onnx_dtype = static_cast<onnx::TensorProto_DataType>(onnx_node_attr.i());
      auto data_type = OnnxNodeParser::GetDataTypeFromOnnx(onnx_dtype);
      (void)prim->AddAttr(ops::kDataType, api::MakeValue(static_cast<int>(data_type)));
    } else if (onnx_node_attr.name() == "shape") {
      std::vector<int64_t> shape;
      std::transform(onnx_node_attr.ints().begin(), onnx_node_attr.ints().end(), std::back_inserter(shape),
                     [](int ele) { return static_cast<int64_t>(ele); });
      (void)prim->AddAttr(ops::kShape, api::MakeValue(shape));
    } else if (onnx_node_attr.name() == "seed") {
      prim->set_seed(static_cast<float>(onnx_node_attr.f()));
    } else if (onnx_node_attr.name() == "mean") {
      prim->set_mean(static_cast<float>(onnx_node_attr.f()));
    } else if (onnx_node_attr.name() == "scale") {
      prim->set_scale(static_cast<float>(onnx_node_attr.f()));
    }
  }
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxRandomNormalParser("RandomNormal", new OnnxRandomNormalParser());
OnnxNodeRegistrar g_onnxRandomNormalLikeParser("RandomNormalLike", new OnnxRandomNormalParser());
}  // namespace lite
}  // namespace mindspore
