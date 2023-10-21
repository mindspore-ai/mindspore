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

#include <memory>
#include <string>
#include <vector>
#include "tools/converter/parser/onnx/onnx_im2col_parser.h"
#include "ops/im2col.h"

namespace mindspore {
namespace lite {
constexpr size_t kDimension1D = 1;
constexpr size_t kDimension2D = 2;
constexpr size_t kAttrSize = 2;
PrimitiveCPtr OnnxIm2ColParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Im2Col>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "dilations") {
      std::vector<int64_t> dilations;
      if (!ParseVecAttr(onnx_node_attr, attribute_name, &dilations)) {
        MS_LOG(ERROR) << "Parse " << attribute_name << " failed.";
        return nullptr;
      }
      prim->set_dilations(dilations);
    } else if (attribute_name == "ksize") {
      std::vector<int64_t> ksizes;
      if (!ParseVecAttr(onnx_node_attr, attribute_name, &ksizes)) {
        MS_LOG(ERROR) << "Parse " << attribute_name << " failed.";
        return nullptr;
      }
      prim->set_ksizes(ksizes);
    } else if (attribute_name == "pads") {
      std::vector<int64_t> pads;
      if (!ParseVecAttr(onnx_node_attr, attribute_name, &pads)) {
        MS_LOG(ERROR) << "Parse " << attribute_name << " failed.";
        return nullptr;
      }
      prim->set_pads(pads);
    } else if (attribute_name == "strides") {
      std::vector<int64_t> strides;
      if (!ParseVecAttr(onnx_node_attr, attribute_name, &strides)) {
        MS_LOG(ERROR) << "Parse " << attribute_name << " failed.";
        return nullptr;
      }
      prim->set_strides(strides);
    }
  }
  return prim->GetPrim();
}

bool OnnxIm2ColParser::ParseVecAttr(const onnx::AttributeProto &onnx_node_attr, const std::string &attr_name,
                                    std::vector<int64_t> *vec) {
  MS_CHECK_TRUE_RET(vec != nullptr, false);
  vec->clear();
  vec->resize(kAttrSize);
  switch (onnx_node_attr.ints().size()) {
    case kDimension1D:
      vec->at(0) = onnx_node_attr.ints(0);
      vec->at(1) = onnx_node_attr.ints(0);
      break;
    case kDimension2D:
      vec->at(0) = onnx_node_attr.ints(0);
      vec->at(1) = onnx_node_attr.ints(1);
      break;
    default:
      MS_LOG(ERROR) << "The dimension of " << attr_name << " should be " << kDimension1D << " or " << kDimension2D
                    << ", but got " << onnx_node_attr.ints().size();
      return false;
  }
  return true;
}

OnnxNodeRegistrar g_onnxIm2ColParser("Im2Col", new OnnxIm2ColParser());
}  // namespace lite
}  // namespace mindspore
