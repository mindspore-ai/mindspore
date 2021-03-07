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

#include "tools/converter/parser/onnx/onnx_constant_parser.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "tools/converter/parser/onnx/onnx_model_parser.h"
#include "ops/constant.h"
#include "src/param_value_lite.h"

namespace mindspore {
namespace lite {
STATUS OnnxConstantParser::AddDataInfoAttr(const onnx::TensorProto &onnx_const_tensor, ops::PrimitiveC *prim) {
  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  if (param_value == nullptr) {
    MS_LOG(ERROR) << "new a paramValueLite failed.";
    return RET_ERROR;
  }
  auto data_type =
    OnnxModelParser::GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(onnx_const_tensor.data_type()));
  if (data_type == kTypeUnknown) {
    MS_LOG(ERROR) << "not support onnx data type "
                  << static_cast<onnx::TensorProto_DataType>(onnx_const_tensor.data_type());
    return RET_ERROR;
  }
  std::vector<int64_t> shape_vector(onnx_const_tensor.dims().begin(), onnx_const_tensor.dims().end());
  std::vector<int> shape;
  std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(shape),
                 [](const int64_t &val) { return static_cast<int32_t>(val); });
  param_value->set_tensor_type(data_type);
  param_value->set_tensor_shape(shape);
  param_value->set_format(schema::Format_NCHW);
  if (OnnxModelParser::CopyOnnxTensorData(onnx_const_tensor, param_value) != RET_OK) {
    MS_LOG(ERROR) << "get value failed.";
    return RET_ERROR;
  }
  prim->set_attr("const_data", param_value);
  return RET_OK;
}

ops::PrimitiveC *OnnxConstantParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Constant>();

  for (const auto &attr : onnx_node.attribute()) {
    if (attr.name() == "sparse_value") {
      MS_LOG(WARNING) << "sparse_value";
      continue;
    }
    if (attr.name() == "value") {
      const auto &const_tensor = attr.t();
      if (AddDataInfoAttr(const_tensor, prim.get()) != RET_OK) {
        MS_LOG(ERROR) << "add basic attr failed.";
        return nullptr;
      }
    } else {
      MS_LOG(ERROR) << "processing Constant op attr " << attr.name() << " not implemented";
      return nullptr;
    }
  }
  return prim.release();
}

OnnxNodeRegistrar g_onnxConstantParser("Constant", new OnnxConstantParser());
}  // namespace lite
}  // namespace mindspore
