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

#include "tools/converter/parser/onnx/onnx_quantize_parser.h"
#include <memory>
#include "ops/quant_dtype_cast.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxQuantizeParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::QuantDTypeCast>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  if (onnx_node.op_type() == "Int8Quantize") {
    prim->set_src_t(kNumberTypeFloat32);
    prim->set_dst_t(kNumberTypeUInt8);
  } else if (onnx_node.op_type() == "Int8Dequantize") {
    prim->set_src_t(kNumberTypeUInt8);
    prim->set_dst_t(kNumberTypeFloat32);
  } else if (onnx_node.op_type() == "AscendDequant") {
    prim->set_src_t(kNumberTypeInt32);
    prim->set_dst_t(kNumberTypeFloat32);
  } else if (onnx_node.op_type() == "AscendQuant") {
    prim->set_src_t(kNumberTypeFloat32);
    prim->set_dst_t(kNumberTypeInt8);
  } else {
    MS_LOG(ERROR) << "Unsupported nodeType: " << onnx_node.op_type().c_str();
    return nullptr;
  }

  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxInt8QuantizeParser("Int8Quantize", new OnnxQuantizeParser());
OnnxNodeRegistrar g_onnxInt8DequantizeParser("Int8Dequantize", new OnnxQuantizeParser());
OnnxNodeRegistrar g_onnxInt8AscendDequantParser("AscendDequant", new OnnxQuantizeParser());
OnnxNodeRegistrar g_onnxInt8AscendQuantParser("AscendQuant", new OnnxQuantizeParser());
}  // namespace lite
}  // namespace mindspore
