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

#include "tools/converter/parser/onnx/onnx_unsqueeze_parser.h"
#include <memory>
#include <vector>
#include "ops/unsqueeze.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxUnSqueezeParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Unsqueeze>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  std::vector<int64_t> axis;
  if (onnx_node.input_size() == 1) {
    for (const auto &onnx_node_attr : onnx_node.attribute()) {
      const auto &attribute_name = onnx_node_attr.name();
      if (attribute_name == "axes") {
        for (int i = 0; i < onnx_node_attr.ints().size(); ++i) {
          axis.emplace_back(onnx_node_attr.ints(i));
        }
      }
    }
  } else {
    MS_CHECK_GE(onnx_node.input_size(), kInputSize1, nullptr);
    const auto &input_name = onnx_node.input(1);
    auto slope_data = OnnxNodeParser::GetConstantTensorData(onnx_graph, input_name);
    if (slope_data == nullptr) {
      MS_LOG(ERROR) << "Failed to find const axis input, input name " << input_name << ", current node "
                    << onnx_node.name();
      return nullptr;
    }
    const auto slope_raw_data = reinterpret_cast<const int64_t *>(slope_data->raw_data().data());
    MS_CHECK_TRUE_RET(slope_raw_data != nullptr, nullptr);
    const int64_t slope_size = slope_data->raw_data().size() / sizeof(int64_t);
    axis.resize(slope_size);
    if (INT_MUL_OVERFLOW_THRESHOLD(slope_size, sizeof(int64_t), SIZE_MAX)) {
      MS_LOG(ERROR) << "data_size overflow";
      return nullptr;
    }
    if (memcpy_s(axis.data(), slope_size * sizeof(int64_t), slope_raw_data, slope_data->raw_data().size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return nullptr;
    }
  }
  prim->set_axis(axis);

  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxUnsqueezeParser("Unsqueeze", new OnnxUnSqueezeParser());
}  // namespace lite
}  // namespace mindspore
