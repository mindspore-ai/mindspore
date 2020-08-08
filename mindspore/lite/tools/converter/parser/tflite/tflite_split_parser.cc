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

#include <vector>
#include <memory>
#include "tools/converter/parser/tflite/tflite_split_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteSplitParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                                const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                                const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                                schema::CNodeT *op,
                                TensorCache *tensor_cache,
                                bool quantizedModel) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  MS_LOG(INFO) << "parse TfliteSplitParser";
  std::unique_ptr<schema::SplitT> attr(new schema::SplitT());

  const auto &tflite_attr = tfliteOp->builtin_options.AsSplitOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name << " attr failed";
    return RET_NULL_PTR;
  }
  auto num_splits = tflite_attr->num_splits;

  const auto &shape_tensor = tfliteTensors[tfliteOp->inputs[1]];
  if (shape_tensor == nullptr) {
    MS_LOG(ERROR) << "shape_tensor is null";
    return RET_NULL_PTR;
  }
  const auto tensor_shape = shape_tensor->shape;
  const auto &axis_tensor = tfliteTensors[tfliteOp->inputs[0]];
  if (axis_tensor == nullptr) {
    MS_LOG(ERROR) << "axis_tensor is null";
    return RET_NULL_PTR;
  }
  auto axis = *(reinterpret_cast<int32_t *>(tfliteModelBuffer[axis_tensor->buffer]->data.data()));
  if (axis < 0) {
    axis += tensor_shape.size();
  }
  if (axis >= tensor_shape.size()) {
    MS_LOG(ERROR) << "axis value too large";
    return RET_ERROR;
  }
  attr->splitDim = axis;
  if (tensor_shape[axis] % num_splits != 0) {
    MS_LOG(ERROR) << "num_splits can't divide tensor's length at axis " << axis;
    return RET_ERROR;
  }
  attr->numberSplit = num_splits;

  for (int i = 0; i < num_splits; i++) {
    attr->sizeSplits.push_back(tensor_shape[axis] / num_splits);
  }

  op->primitive->value.type = schema::PrimitiveType_Split;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

TfliteNodeRegister g_TfliteSplitParser("Split", new TfliteSplitParser());
}  // namespace lite
}  // namespace mindspore
