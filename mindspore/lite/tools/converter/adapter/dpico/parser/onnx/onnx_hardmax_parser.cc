/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "parser/onnx/onnx_hardmax_parser.h"
#include <memory>
#include <string>
#include "common/op_attr.h"
#include "ops/custom.h"
#include "./onnx.pb.h"
#include "include/registry/node_parser_registry.h"
#include "tools/converter/parser/onnx/onnx_node_parser.h"
#include "ops/auto_generate/gen_lite_ops.h"

namespace mindspore {
namespace lite {
ops::BaseOperatorPtr OnnxHardmaxParser::Parse(const onnx::GraphProto &onnx_proto, const onnx::NodeProto &onnx_node) {
  auto prim = api::MakeShared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Custom prim failed.";
    return nullptr;
  }
  prim->set_type("Hardmax");
  int64_t axis = 0;
  bool axis_is_def = true;
  auto iter = std::find_if(onnx_node.attribute().begin(), onnx_node.attribute().end(),
                           [](const onnx::AttributeProto onnx_node_attr) { return onnx_node_attr.name() == "axis"; });
  if (iter != onnx_node.attribute().end()) {
    axis = static_cast<int64_t>((*iter).i());
    axis_is_def = false;
  }
  if (axis_is_def) {
    axis = OnnxNodeParser::opset_version() >= 13 ? -1 : 1;
  }
  // set attr for mapper
  (void)prim->AddAttr("axis", api::MakeValue(axis));

  return prim;
}
}  // namespace lite
}  // namespace mindspore
using mindspore::converter::kFmkTypeOnnx;
namespace mindspore::registry {
REG_NODE_PARSER(kFmkTypeOnnx, Hardmax, std::make_shared<lite::OnnxHardmaxParser>())
}  // namespace mindspore::registry
