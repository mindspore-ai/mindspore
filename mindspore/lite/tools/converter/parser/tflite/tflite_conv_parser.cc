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

#include "tools/converter/parser/tflite/tflite_conv_parser.h"
#include <vector>
#include <memory>
#include <map>

namespace mindspore {
namespace lite {
STATUS TfliteConvParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                               const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                               const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                               schema::CNodeT *op,
                               std::vector<int32_t> *tensors_id,
                               std::vector<schema::Format> *tensors_format,
                               std::map<int, int>  *tensors_id_map) {
  MS_LOG(DEBUG) << "parse TfliteConvParser";

  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::Conv2DT> attr(new schema::Conv2DT());
  const auto &tflite_attr = tflite_op->builtin_options.AsConv2DOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
    return RET_NULL_PTR;
  }
  attr->group = 1;
  attr->strideW = tflite_attr->stride_w;
  attr->strideH = tflite_attr->stride_h;
  attr->dilateH = tflite_attr->dilation_h_factor;
  attr->dilateW = tflite_attr->dilation_w_factor;
  attr->padMode = GetPadMode(tflite_attr->padding);
  attr->format = schema::Format_NHWC;
  attr->activationType = GetActivationFunctionType(tflite_attr->fused_activation_function);
  attr->hasBias = true;

  // get the conv op weight tensor
  auto weight_index = tflite_op->inputs[1];
  const auto &weight_tensor = tflite_tensors[weight_index];
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "the weight tensor is null";
    return RET_NULL_PTR;
  }
  auto weight_shape = weight_tensor->shape;
  attr->channelIn = weight_shape[3];
  attr->channelOut = weight_shape[0];
  attr->kernelH = weight_shape[1];
  attr->kernelW = weight_shape[2];

  // calculate pad params
  auto data_index = tflite_op->inputs[0];
  const auto &data_tensor = tflite_tensors[data_index];
  std::vector<int> params;
  if (getPaddingParam(data_tensor, attr->padMode, attr->strideH,
                      attr->strideW, attr->kernelH, attr->kernelW, &params) != RET_OK) {
    MS_LOG(ERROR) << "get padding params failed";
    return RET_ERROR;
  } else {
    attr->padUp = params.at(0);
    attr->padDown = params.at(1);
    attr->padLeft = params.at(2);
    attr->padRight = params.at(3);
  }

  op->primitive->value.type = schema::PrimitiveType_Conv2D;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_id, tensors_format, tensors_id_map,
             tflite_op->inputs[0], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  AddOpInput(op, tensors_id, tensors_format, tensors_id_map,
             tflite_op->inputs[1], tensors_id->size(), tflite_tensors.size(), schema::Format_KHWC);
  AddOpInput(op, tensors_id, tensors_format, tensors_id_map,
             tflite_op->inputs[2], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  AddOpOutput(op, tensors_id, tensors_format, tensors_id_map,
              tflite_op->outputs[0], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteConv2DParser("Conv2D", new TfliteConvParser());
}  // namespace lite
}  // namespace mindspore


