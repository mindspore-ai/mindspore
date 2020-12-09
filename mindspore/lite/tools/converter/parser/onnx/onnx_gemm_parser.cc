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

namespace mindspore {
namespace lite {
lite::PrimitiveC *OnnxGemmParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                     const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx IdentityParser";
  auto node_parser = OnnxNodeParserRegistry::GetInstance()->GetNodeParser("MatMul");
  if (node_parser == nullptr) {
    MS_LOG(ERROR) << "parse node " << onnx_node.op_type() << " failed.";
    return nullptr;
  }
  auto *matmul_primitive = node_parser->ParseLitePrimitive(onnx_graph, onnx_node);

  node_parser = OnnxNodeParserRegistry::GetInstance()->GetNodeParser("BiasAdd");
  if (node_parser == nullptr) {
    MS_LOG(ERROR) << "parse node " << onnx_node.op_type() << " failed.";
    return nullptr;
  }

  auto *bias_add_primitive = node_parser->ParseLitePrimitive(onnx_graph, onnx_node);
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }

  primitive->value.type = schema::PrimitiveType_MakeTuple;
  auto primitve_c = PrimitiveC::Create(primitive.release());
  primitve_c->set_attr("MatMul", std::shared_ptr<lite::PrimitiveC>(matmul_primitive));
  primitve_c->set_attr("BiasAdd", std::shared_ptr<lite::PrimitiveC>(bias_add_primitive));
  return primitve_c;
}

OnnxNodeRegistrar g_onnxGemmParser("Gemm", new OnnxGemmParser());
}  // namespace lite
}  // namespace mindspore
