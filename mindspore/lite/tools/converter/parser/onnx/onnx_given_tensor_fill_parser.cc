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

#include "tools/converter/parser/onnx/onnx_given_tensor_fill_parser.h"
#include <functional>
#include <memory>
#include <vector>
#include <algorithm>
#include "src/param_value_lite.h"
#include "ops/constant.h"

namespace mindspore {
namespace lite {
STATUS OnnxGivenTensorFillParser::ParseInt8GivenIntTensorFill(const onnx::NodeProto &onnx_node, ops::PrimitiveC *prim,
                                                              const std::vector<int> &shape) {
  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();

  int data_count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto iter = std::find_if(onnx_node.attribute().begin(), onnx_node.attribute().end(),
                           [](const onnx::AttributeProto &attr) { return attr.name() == "values"; });
  if (iter == onnx_node.attribute().end()) {
    return RET_OK;
  }
  size_t data_size = data_count * sizeof(int64_t) / sizeof(uint8_t);
  char *param_data = new (std::nothrow) char[data_size];
  if (param_data == nullptr) {
    MS_LOG(ERROR) << "new char[] failed";
    return RET_MEMORY_FAILED;
  }
  if (iter->ints().data() == nullptr) {
    MS_LOG(ERROR) << "origin ints data in onnx is nullptr";
    delete[] param_data;
    return RET_NULL_PTR;
  }
  if (memcpy_s(param_data, data_size, iter->ints().data(), data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    delete[] param_data;
    return RET_ERROR;
  }
  param_value->set_tensor_shape(shape);
  param_value->set_format(schema::Format_NUM_OF_FORMAT);
  param_value->set_tensor_type(kNumberTypeInt64);
  param_value->SetTensorData(param_data, data_size);
  prim->set_attr("const_data", param_value);
  return RET_OK;
}

STATUS OnnxGivenTensorFillParser::ParseInt8GivenTensorFill(const onnx::NodeProto &onnx_node, ops::PrimitiveC *prim,
                                                           const std::vector<int> &shape) {
  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();

  int data_count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  auto iter = std::find_if(onnx_node.attribute().begin(), onnx_node.attribute().end(),
                           [](const onnx::AttributeProto &attr) { return attr.name() == "values"; });
  if (iter == onnx_node.attribute().end()) {
    return RET_OK;
  }
  char *param_data = new (std::nothrow) char[data_count];
  if (param_data == nullptr) {
    MS_LOG(ERROR) << "new char[] failed";
    return RET_MEMORY_FAILED;
  }
  if (memcpy_s(param_data, data_count, iter->s().data(), data_count) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    delete[] param_data;
    return RET_ERROR;
  }
  param_value->set_tensor_shape(shape);
  param_value->set_format(schema::Format_NUM_OF_FORMAT);
  param_value->set_tensor_type(kNumberTypeUInt8);
  param_value->SetTensorData(param_data, data_count);
  prim->set_attr("const_data", param_value);
  return RET_OK;
}
ops::PrimitiveC *OnnxGivenTensorFillParser::Parse(const onnx::GraphProto &onnx_graph,
                                                  const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Constant>();

  std::vector<int64_t> shape_vector;
  auto iter = std::find_if(onnx_node.attribute().begin(), onnx_node.attribute().end(),
                           [](const onnx::AttributeProto &attr) { return attr.name() == "shape"; });
  if (iter != onnx_node.attribute().end()) {
    shape_vector.insert(shape_vector.begin(), iter->ints().begin(), iter->ints().end());
  }
  std::vector<int> shape;
  std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(shape),
                 [](const int64_t &val) { return static_cast<int32_t>(val); });
  if (onnx_node.op_type() == "Int8GivenIntTensorFill") {
    if (ParseInt8GivenIntTensorFill(onnx_node, prim.get(), shape) != RET_OK) {
      MS_LOG(ERROR) << "given tensor fill parse failed.";
      return nullptr;
    }
  } else if (onnx_node.op_type() == "Int8GivenTensorFill") {
    if (ParseInt8GivenTensorFill(onnx_node, prim.get(), shape) != RET_OK) {
      MS_LOG(ERROR) << "given tensor fill parse failed.";
      return nullptr;
    }
  }

  return prim.release();
}

OnnxNodeRegistrar g_onnxInt8GivenIntTensorFillParser("Int8GivenIntTensorFill", new OnnxGivenTensorFillParser());
OnnxNodeRegistrar g_onnxInt8GivenTensorFillParser("Int8GivenTensorFill", new OnnxGivenTensorFillParser());
}  // namespace lite
}  // namespace mindspore
