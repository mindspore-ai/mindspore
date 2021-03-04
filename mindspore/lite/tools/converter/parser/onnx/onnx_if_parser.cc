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

#include "tools/converter/parser/onnx/onnx_if_parser.h"
#include <memory>
#include "tools/converter/parser/onnx/onnx_model_parser.h"
#include "tools/converter/ops/ops_def.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxIfParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<If>();
  return prim.release();
}

OnnxNodeRegistrar g_onnxIfParser("If", new OnnxIfParser());
}  // namespace lite
}  // namespace mindspore
