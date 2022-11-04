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

#include "parser/onnx/onnx_hardsigmoid_parser.h"
#include <memory>
#include <string>
#include "common/op_attr.h"
#include "ops/custom.h"
#include "./onnx.pb.h"
#include "include/registry/node_parser_registry.h"
#include "mindapi/base/logging.h"

namespace mindspore {
namespace lite {
ops::BaseOperatorPtr OnnxHardSigmoidParser::Parse(const onnx::GraphProto &onnx_proto,
                                                  const onnx::NodeProto &onnx_node) {
  auto prim = api::MakeShared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Custom prim failed.";
    return nullptr;
  }
  prim->set_type("HardSigmoid");
  float alpha = 0.2;
  float beta = 0.5;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "alpha") {
      alpha = static_cast<float>(onnx_node_attr.f());
    } else if (onnx_node_attr.name() == "beta") {
      beta = static_cast<float>(onnx_node_attr.f());
    }
  }
  // set attr for mapper
  (void)prim->AddAttr("alpha", api::MakeValue(alpha));
  (void)prim->AddAttr("beta", api::MakeValue(beta));

  return prim;
}
}  // namespace lite
}  // namespace mindspore
using mindspore::converter::kFmkTypeOnnx;
namespace mindspore::registry {
REG_NODE_PARSER(kFmkTypeOnnx, HardSigmoid, std::make_shared<lite::OnnxHardSigmoidParser>())
}  // namespace mindspore::registry
