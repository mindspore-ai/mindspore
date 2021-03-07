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

#include "tools/converter/parser/onnx/onnx_identity_parser.h"
#include <vector>
#include <memory>
#include "ops/identity.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxIdentityParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Identity>();
  return prim.release();
}

OnnxNodeRegistrar g_onnxIdentityParser("Identity", new OnnxIdentityParser());
}  // namespace lite
}  // namespace mindspore
