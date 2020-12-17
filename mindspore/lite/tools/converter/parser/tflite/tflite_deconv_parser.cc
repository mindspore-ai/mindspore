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

#include "tools/converter/parser/tflite/tflite_deconv_parser.h"
#include <vector>
#include <memory>

namespace mindspore::lite {
PrimitiveC *TfliteDeConvParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                   const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto primitive = std::make_unique<schema::PrimitiveT>();
  auto &tflite_subgraph = tflite_model->subgraphs.front();
  std::unique_ptr<schema::DeConv2DT> attr = std::make_unique<schema::DeConv2DT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsTransposeConvOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op deconv attr failed";
    return nullptr;
  }

  attr->group = 1;
  attr->strideW = tflite_attr->stride_w;
  attr->strideH = tflite_attr->stride_h;
  attr->dilateH = 1;
  attr->dilateW = 1;
  attr->padMode = GetPadMode(tflite_attr->padding);
  attr->format = schema::Format::Format_NHWC;
  attr->activationType = schema::ActivationType_NO_ACTIVATION;

  // get the conv op weight tensor
  auto weight_index = tflite_op->inputs[1];
  const auto &weight_tensor = tflite_subgraph->tensors[weight_index];
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "the weight tensor is null";
    return nullptr;
  }
  auto weight_shape = weight_tensor->shape;
  attr->channelIn = weight_shape[3];
  attr->channelOut = weight_shape[0];
  attr->kernelH = weight_shape[1];
  attr->kernelW = weight_shape[2];

  // calculate pad params
  auto data_index = tflite_op->inputs[2];
  const auto &data_tensor = tflite_subgraph->tensors[data_index];
  std::vector<int64_t> params;
  int status =
    getPaddingParam(data_tensor, attr->padMode, attr->strideH, attr->strideW, attr->kernelH, attr->kernelW, &params);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get padding params failed";
    return nullptr;
  } else if (status == RET_OK) {
    attr->padUp = params.at(0);
    attr->padDown = params.at(1);
    attr->padLeft = params.at(2);
    attr->padRight = params.at(3);
  }
  primitive->value.type = schema::PrimitiveType_DeConv2D;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteDeConv2DParser(tflite::BuiltinOperator_TRANSPOSE_CONV, new TfliteDeConvParser());
}  // namespace mindspore::lite
