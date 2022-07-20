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

#include "tools/converter/parser/onnx/onnx_quantize_linear_parser.h"
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <utility>
#include <cstdint>
#include "nnacl/op_base.h"
#include "tools/converter/ops/ops_def.h"
#include "ops/op_utils.h"

namespace mindspore::lite {
const void *OnnxQuantizeLinearParser::GetConstData(const onnx::GraphProto &onnx_graph, const std::string &input_name,
                                                   TypeId *data_type) {
  auto node_iter = std::find_if(onnx_graph.initializer().begin(), onnx_graph.initializer().end(),
                                [input_name](const onnx::TensorProto &proto) { return proto.name() == input_name; });
  if (node_iter == onnx_graph.initializer().end()) {
    MS_LOG(ERROR) << "graph not find node: " << input_name;
    return nullptr;
  }
  bool overflow = false;
  auto data_count = OnnxNodeParser::GetOnnxElementNum((*node_iter), &overflow);

  size_t data_size = 0;
  *data_type = OnnxNodeParser::GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>((*node_iter).data_type()));
  auto onnx_data = OnnxNodeParser::GetOnnxRawData((*node_iter), *data_type, data_count, &data_size);
  return onnx_data;
}

PrimitiveCPtr OnnxQuantizeLinearParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<QuantizeLinear>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  MS_CHECK_GE(onnx_node.input_size(), kInputSize2, nullptr);
  const auto &onnx_quantize_scale = onnx_node.input(SECOND_INPUT);
  const auto &onnx_quantize_zero_point = onnx_node.input(THIRD_INPUT);

  // scale attr
  TypeId scale_data_type;
  auto onnx_scale_data = GetConstData(onnx_graph, onnx_quantize_scale, &scale_data_type);
  if (onnx_scale_data == nullptr) {
    MS_LOG(ERROR) << "Onnx scale data is nullptr.";
    return nullptr;
  }
  float scale = *(static_cast<const float *>(onnx_scale_data));
  prim->AddAttr("scale", MakeValue(scale));

  // zero_point attr
  TypeId zp_data_type = mindspore::kTypeUnknown;
  auto onnx_zero_point_data = GetConstData(onnx_graph, onnx_quantize_zero_point, &zp_data_type);
  if (onnx_zero_point_data == nullptr) {
    MS_LOG(ERROR) << "Onnx zero point data is nullptr.";
    return nullptr;
  }
  int zero_point = 0;
  if (zp_data_type == mindspore::kNumberTypeUInt8) {
    zero_point = *(static_cast<const uint8_t *>(onnx_zero_point_data)) - 128;
  } else if (zp_data_type == mindspore::kNumberTypeInt8) {
    zero_point = *(static_cast<const int *>(onnx_zero_point_data));
  } else {
    MS_LOG(ERROR) << "Invalid zero point data type: " << zp_data_type;
  }
  prim->AddAttr("zero_point", MakeValue(zero_point));

  return prim;
}

OnnxNodeRegistrar g_onnxQuantizeLinearParser("QuantizeLinear", new OnnxQuantizeLinearParser());
}  // namespace mindspore::lite
