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
#include <algorithm>
#include <memory>
#include <vector>
#include "tools/common/tensor_util.h"
#include "tools/converter/ops/ops_def.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
STATUS ParseInt8GivenIntTensorFill(const onnx::NodeProto &onnx_node, ops::PrimitiveC *prim,
                                   const std::vector<int> &shape) {
  MS_ASSERT(prim != nullptr);
  int data_count = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(data_count, shape.at(i)), RET_ERROR, "Int mul overflow.");
    data_count = data_count * shape.at(i);
  }
  auto iter = std::find_if(onnx_node.attribute().begin(), onnx_node.attribute().end(),
                           [](const onnx::AttributeProto &attr) { return attr.name() == "values"; });
  if (iter == onnx_node.attribute().end()) {
    return RET_OK;
  }
  ShapeVector shape_vector(shape.begin(), shape.end());
  if (INT_MUL_OVERFLOW_THRESHOLD(data_count, sizeof(int64_t), SIZE_MAX)) {
    MS_LOG(ERROR) << "data_size overflow";
    return RET_ERROR;
  }
  size_t data_size = data_count * sizeof(int64_t) / sizeof(uint8_t);
  auto tensor_info = CreateTensorInfo(iter->ints().data(), data_size, shape_vector, kNumberTypeInt64);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return RET_ERROR;
  }
  prim->set_attr("const_data", tensor_info);
  return RET_OK;
}

STATUS ParseInt8GivenTensorFill(const onnx::NodeProto &onnx_node, ops::PrimitiveC *prim,
                                const std::vector<int> &shape) {
  MS_ASSERT(prim != nullptr);
  int data_count = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(data_count, shape.at(i)), RET_ERROR, "Int mul overflow.");
    data_count = data_count * shape.at(i);
  }
  auto iter = std::find_if(onnx_node.attribute().begin(), onnx_node.attribute().end(),
                           [](const onnx::AttributeProto &attr) { return attr.name() == "values"; });
  if (iter == onnx_node.attribute().end()) {
    return RET_OK;
  }
  ShapeVector shape_vector(shape.begin(), shape.end());
  auto tensor_info = CreateTensorInfo(iter->s().data(), data_count, shape_vector, kNumberTypeUInt8);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return RET_ERROR;
  }
  prim->set_attr("const_data", tensor_info);
  return RET_OK;
}
}  // namespace

ops::PrimitiveC *OnnxGivenTensorFillParser::Parse(const onnx::GraphProto &onnx_graph,
                                                  const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<lite::Constant>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
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
