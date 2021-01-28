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

#include "tools/converter/parser/onnx/onnx_gemm_parser.h"
#include <vector>
#include <memory>
#include "ops/make_tuple.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxGemmParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::MakeTuple>();

  auto node_parser = OnnxNodeParserRegistry::GetInstance()->GetNodeParser("MatMul");
  if (node_parser == nullptr) {
    MS_LOG(ERROR) << "parse node " << onnx_node.op_type() << " failed.";
    return nullptr;
  }
  auto *matmul_primitive = node_parser->Parse(onnx_graph, onnx_node);
  prim->AddAttr("MatMul", std::shared_ptr<ops::PrimitiveC>(matmul_primitive));

  node_parser = OnnxNodeParserRegistry::GetInstance()->GetNodeParser("BiasAdd");
  if (node_parser == nullptr) {
    MS_LOG(ERROR) << "parse node " << onnx_node.op_type() << " failed.";
    return nullptr;
  }
  auto *bias_add_primitive = node_parser->Parse(onnx_graph, onnx_node);
  prim->AddAttr("BiasAdd", std::shared_ptr<ops::PrimitiveC>(bias_add_primitive));

  return prim.release();
}

OnnxNodeRegistrar g_onnxGemmParser("Gemm", new OnnxGemmParser());
}  // namespace lite
}  // namespace mindspore
