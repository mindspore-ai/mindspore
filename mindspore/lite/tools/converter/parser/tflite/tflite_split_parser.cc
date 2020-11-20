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

#include "tools/converter/parser/tflite/tflite_split_parser.h"
#include <vector>
#include <memory>
#include <map>

namespace mindspore {
namespace lite {
STATUS TfliteSplitParser::Parse(TfliteTensorsInfo *tensors_info, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                const std::unique_ptr<tflite::ModelT> &tflite_model,
                                const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "parse TfliteSplitParser";
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

  const auto &tflite_attr = tflite_op->builtin_options.AsSplitOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name << " attr failed";
    return RET_NULL_PTR;
  }
  auto num_splits = tflite_attr->num_splits;

  const auto &shape_tensor = tflite_subgraph->tensors[tflite_op->inputs[1]];
  if (shape_tensor == nullptr) {
    MS_LOG(ERROR) << "shape_tensor is null";
    return RET_NULL_PTR;
  }
  const auto tensor_shape = shape_tensor->shape;

  const auto &axis_tensor = tflite_subgraph->tensors[tflite_op->inputs[0]];
  if (axis_tensor == nullptr) {
    MS_LOG(ERROR) << "axis_tensor is null";
    return RET_NULL_PTR;
  }
  auto axis = *(reinterpret_cast<int32_t *>(tflite_model->buffers[axis_tensor->buffer]->data.data()));
  if (axis < 0) {
    axis += tensor_shape.size();
  }
  if (axis >= static_cast<int>(tensor_shape.size())) {
    MS_LOG(ERROR) << "axis value is too large";
    return RET_ERROR;
  }
  attr->splitDim = axis;
  if (num_splits == 0) {
    MS_LOG(ERROR) << "Divide-by-zero error!";
    return RET_ERROR;
  }
  if (tensor_shape[axis] % num_splits != 0 && tensor_shape[axis] / num_splits != 0) {
    MS_LOG(ERROR) << "num_splits can't divide tensor's length at axis " << axis;
    return RET_ERROR;
  }
  attr->numberSplit = num_splits;
  if (tensor_shape[axis] / num_splits != 0) {
    for (int i = 0; i < num_splits; i++) {
      attr->sizeSplits.push_back(tensor_shape[axis] / num_splits);
    }
  }

  op->primitive->value.type = schema::PrimitiveType_Split;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_info, tflite_op->inputs[1], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  for (int output : tflite_op->outputs) {
    AddOpOutput(op, tensors_info, output, tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  }
  return RET_OK;
}

TfliteNodeRegister g_tfliteSplitParser("Split", new TfliteSplitParser());
}  // namespace lite
}  // namespace mindspore
