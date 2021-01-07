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

#include "tools/converter/parser/onnx/onnx_slice_parser.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>
#include <string>

namespace mindspore {
namespace lite {
lite::PrimitiveC *OnnxSliceParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                      const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx SliceParser";
  auto attr = std::make_unique<schema::StridedSliceT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  std::vector<int> starts;
  std::vector<int> ends;
  std::vector<int> axes;
  std::vector<int> steps;
  constexpr int64_t int_32_max = INT32_MAX;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "starts") {
      const int num = onnx_node_attr.ints_size();
      starts.clear();
      for (int i = 0; i < num; ++i) {
        starts.push_back(static_cast<int>(std::min(onnx_node_attr.ints()[i], int_32_max)));
      }
    } else if (attribute_name == "axes") {
      const int num = onnx_node_attr.ints_size();
      axes.clear();
      for (int i = 0; i < num; ++i) {
        axes.push_back(static_cast<int>(std::min(onnx_node_attr.ints()[i], int_32_max)));
      }
    } else if (attribute_name == "ends") {
      const int num = onnx_node_attr.ints_size();
      ends.clear();
      for (int i = 0; i < num; ++i) {
        ends.push_back(static_cast<int>(std::min(onnx_node_attr.ints()[i], int_32_max)));
      }
    } else if (attribute_name == "steps") {
      const int num = onnx_node_attr.ints_size();
      steps.clear();
      for (int i = 0; i < num; ++i) {
        steps.push_back(static_cast<int>(std::min(onnx_node_attr.ints()[i], int_32_max)));
      }
    }
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_StridedSlice;
  primitive->value.value = attr.release();
  auto primitive_c = PrimitiveC::Create(primitive.release());
  if (starts.empty()) {
    return primitive_c;
  }
  if (axes.empty()) {
    for (size_t i = 0; i < starts.size(); ++i) {
      axes.push_back(i);
    }
  }
  if (steps.empty()) {
    steps.assign(starts.size(), 1);
  }
  primitive_c->set_attr("starts", MakeValue<std::vector<int>>(starts));
  primitive_c->set_attr("ends", MakeValue<std::vector<int>>(ends));
  primitive_c->set_attr("axes", MakeValue<std::vector<int>>(axes));
  primitive_c->set_attr("steps", MakeValue<std::vector<int>>(steps));
  return primitive_c;
}

OnnxNodeRegistrar g_onnxSliceParser("Slice", new OnnxSliceParser());
}  // namespace lite
}  // namespace mindspore
