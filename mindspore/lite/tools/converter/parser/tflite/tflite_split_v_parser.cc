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

#include "tools/converter/parser/tflite/tflite_split_v_parser.h"
#include <vector>
#include <memory>
#include <map>

namespace mindspore {
namespace lite {
STATUS TfliteSplitVParser::Parse(TfliteTensorsInfo *tensors_info, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                 const std::unique_ptr<tflite::ModelT> &tflite_model,
                                 const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "parse TfliteSplitVParser";
  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  MS_ASSERT(tflite_subgraph != nullptr);
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::SplitT> attr = std::make_unique<schema::SplitT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsSplitVOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name << " attr failed";
    return RET_NULL_PTR;
  }
  attr->numberSplit = tflite_attr->num_splits;

  if (GetTfliteData(tflite_op->inputs[1], tflite_subgraph->tensors, tflite_model->buffers, attr->sizeSplits)) {
    MS_LOG(ERROR) << "get spliteV -> sizeSplits failed";
    return RET_ERROR;
  }

  const auto &tensor = tflite_subgraph->tensors[tflite_op->inputs[0]];
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "tensor_shape is null";
    return RET_NULL_PTR;
  }
  auto tensor_shape = tensor->shape;
  const auto &axis_tensor = tflite_subgraph->tensors[tflite_op->inputs[2]];
  if (axis_tensor == nullptr) {
    MS_LOG(ERROR) << "axis_tensor is null";
    return RET_NULL_PTR;
  }
  const auto &axis_buf = tflite_model->buffers[axis_tensor->buffer];
  if (axis_buf == nullptr) {
    MS_LOG(ERROR) << "axis_buf is null";
    return RET_NULL_PTR;
  }
  auto axis = *(reinterpret_cast<int32_t *>(axis_buf->data.data()));
  if (axis < 0) {
    axis += tensor_shape.size();
  }
  if (axis >= static_cast<int>(tensor_shape.size())) {
    MS_LOG(ERROR) << "axis value is too large";
    return RET_ERROR;
  }
  attr->splitDim = axis;

  op->primitive->value.type = schema::PrimitiveType_Split;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_info, tflite_op->inputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  for (int output : tflite_op->outputs) {
    AddOpOutput(op, tensors_info, output, tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  }
  return RET_OK;
}
PrimitiveC *TfliteSplitVParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                   const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto &tflite_subgraph = tflite_model->subgraphs.front();
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is null";
    return nullptr;
  }

  std::unique_ptr<schema::SplitT> attr = std::make_unique<schema::SplitT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsSplitVOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op splitv attr failed";
    return nullptr;
  }
  attr->numberSplit = tflite_attr->num_splits;

  if (GetTfliteData(tflite_op->inputs[1], tflite_subgraph->tensors, tflite_model->buffers, attr->sizeSplits)) {
    MS_LOG(ERROR) << "get spliteV -> sizeSplits failed";
    return nullptr;
  }

  const auto &tensor = tflite_subgraph->tensors[tflite_op->inputs[0]];
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "tensor_shape is null";
    return nullptr;
  }
  auto tensor_shape = tensor->shape;
  const auto &axis_tensor = tflite_subgraph->tensors[tflite_op->inputs[2]];
  if (axis_tensor == nullptr) {
    MS_LOG(ERROR) << "axis_tensor is null";
    return nullptr;
  }
  auto axis = *(reinterpret_cast<int32_t *>(tflite_model->buffers[axis_tensor->buffer]->data.data()));
  if (axis < 0) {
    axis += tensor_shape.size();
  }
  if (axis >= static_cast<int>(tensor_shape.size())) {
    MS_LOG(ERROR) << "axis value is too large";
    return nullptr;
  }
  attr->splitDim = axis;

  primitive->value.type = schema::PrimitiveType_Split;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteSplitVParser("SplitV", new TfliteSplitVParser());
}  // namespace lite
}  // namespace mindspore
