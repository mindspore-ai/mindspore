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

#include "parser/onnx/onnx_shrink_parser.h"
#include <memory>
#include <string>
#include "common/op_attr.h"
#include "ops/custom.h"
#include "./onnx.pb.h"
#include "include/registry/node_parser_registry.h"
#include "mindapi/base/logging.h"

namespace mindspore {
namespace lite {
ops::BaseOperatorPtr OnnxShrinkParser::Parse(const onnx::GraphProto &onnx_proto, const onnx::NodeProto &onnx_node) {
  auto prim = api::MakeShared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Custom prim failed.";
    return nullptr;
  }
  prim->set_type("Shrink");
  float bias = 0.0;
  float lambd = 0.5;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "bias") {
      bias = onnx_node_attr.f();
    } else if (onnx_node_attr.name() == "lambd") {
      lambd = onnx_node_attr.f();
    }
  }
  // set attr for mapper
  (void)prim->AddAttr("bias", api::MakeValue(bias));
  (void)prim->AddAttr("lambd", api::MakeValue(lambd));

  return prim;
}
}  // namespace lite
}  // namespace mindspore
using mindspore::converter::kFmkTypeOnnx;
namespace mindspore::registry {
REG_NODE_PARSER(kFmkTypeOnnx, Shrink, std::make_shared<lite::OnnxShrinkParser>())
}  // namespace mindspore::registry
