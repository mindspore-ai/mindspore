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

#include "tools/converter/parser/tflite/tflite_argmax_parser.h"
#include <memory>
#include <vector>
#include <map>

namespace mindspore {
namespace lite {
STATUS TfliteArgmaxParser::Parse(TfliteTensorsInfo *tensors_info, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                 const std::unique_ptr<tflite::ModelT> &tflite_model,
                                 const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "parse TfliteArgmaxParser";
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

  std::unique_ptr<schema::ArgMaxT> attr = std::make_unique<schema::ArgMaxT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  attr->outMaxValue = false;
  attr->topK = 1;
  attr->keepDims = false;
  attr->axisType = 1;

  // get axis attr
  auto axis_idx = tflite_op->inputs[1];
  auto axis_tensor = tflite_subgraph->tensors[axis_idx].get();
  if (axis_tensor == nullptr) {
    MS_LOG(ERROR) << "axis_tensor is null";
    return RET_NULL_PTR;
  }
  auto buffer_idx = axis_tensor->buffer;
  auto &buf_data = tflite_model->buffers[buffer_idx];
  if (buf_data == nullptr) {
    MS_LOG(ERROR) << "the buf data is null";
    return RET_NULL_PTR;
  }
  auto data_ptr = buf_data->data.data();
  if (data_ptr == nullptr) {
    MS_LOG(ERROR) << "the data is null";
    return RET_NULL_PTR;
  }
  attr->axis = *(static_cast<int32_t *>(static_cast<void *>(data_ptr)));

  op->primitive->value.type = schema::PrimitiveType_ArgMax;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_info, tflite_op->inputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  AddOpOutput(op, tensors_info, tflite_op->outputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}
PrimitiveC *TfliteArgmaxParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                   const std::unique_ptr<tflite::ModelT> &tflite_model) {
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  std::unique_ptr<schema::ArgMaxT> attr = std::make_unique<schema::ArgMaxT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  attr->outMaxValue = false;
  attr->topK = 1;
  attr->keepDims = false;
  attr->axisType = 1;

  // get axis attr
  auto axis_idx = tflite_op->inputs[1];
  auto buffer_idx = tflite_subgraph->tensors[axis_idx]->buffer;
  auto &buf_data = tflite_model->buffers[buffer_idx];
  if (buf_data == nullptr) {
    MS_LOG(ERROR) << "the buf data is null";
    return nullptr;
  }
  auto data_ptr = buf_data->data.data();
  if (data_ptr == nullptr) {
    MS_LOG(ERROR) << "the data is null";
    return nullptr;
  }
  attr->axis = *(static_cast<int32_t *>(static_cast<void *>(data_ptr)));
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_ArgMax;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteArgmaxParser("Argmax", new TfliteArgmaxParser());
}  // namespace lite
}  // namespace mindspore
