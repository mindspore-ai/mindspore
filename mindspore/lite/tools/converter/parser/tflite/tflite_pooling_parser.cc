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

#include "tools/converter/parser/tflite/tflite_pooling_parser.h"
#include <vector>
#include <memory>
#include <string>
#include <map>

namespace mindspore {
namespace lite {
STATUS TflitePoolingParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                  const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                                  const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                  schema::CNodeT *op,
                                  std::vector<int32_t> *tensors_id,
                                  std::vector<schema::Format> *tensors_format,
                                  std::map<int, int>  *tensors_id_map) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::PoolingT> attr = std::make_unique<schema::PoolingT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  std::vector<std::string> node_name_str;
  Split(op->name, &node_name_str, "-");
  const char *node_name = node_name_str.data()->c_str();
  if (std::strcmp(node_name, "MeanPooling") == 0) {
    MS_LOG(DEBUG) << "parser TfliteMeanPoolingParser";
    attr->poolingMode = schema::PoolMode_MEAN_POOLING;
  } else if (std::strcmp(node_name, "MaxPooling") == 0) {
    MS_LOG(DEBUG) << "parse TfliteMaxPoolingParser";
    attr->poolingMode = schema::PoolMode_MAX_POOLING;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsPool2DOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
    return RET_NULL_PTR;
  }
  attr->windowW = tflite_attr->filter_width;
  attr->windowH = tflite_attr->filter_height;
  attr->strideW = tflite_attr->stride_w;
  attr->strideH = tflite_attr->stride_h;
  attr->padMode = GetPadMode(tflite_attr->padding);
  attr->format = schema::Format_NHWC;

  attr->global = false;
  attr->roundMode = schema::RoundMode_FLOOR;
  attr->activationType = GetActivationFunctionType(tflite_attr->fused_activation_function);

  // calculate pad params
  auto data_index = tflite_op->inputs[0];
  const auto &data_tensor = tflite_tensors[data_index];
  std::vector<int> params;
  if (getPaddingParam(data_tensor, attr->padMode, attr->strideH,
                      attr->strideW, attr->windowH, attr->windowW, &params) != RET_OK) {
    MS_LOG(ERROR) << "get padding params failed";
    return RET_ERROR;
  } else {
    attr->padUp = params.at(0);
    attr->padDown = params.at(1);
    attr->padLeft = params.at(2);
    attr->padRight = params.at(3);
  }

  op->primitive->value.type = schema::PrimitiveType_Pooling;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_id, tensors_format, tensors_id_map,
             tflite_op->inputs[0], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  AddOpOutput(op, tensors_id, tensors_format, tensors_id_map,
              tflite_op->outputs[0], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteMeanPoolingParser("MeanPooling", new TfliteMeanPoolingParser());
TfliteNodeRegister g_tfliteMaxPoolingParser("MaxPooling", new TfliteMaxPoolingParser());
}  // namespace lite
}  // namespace mindspore


