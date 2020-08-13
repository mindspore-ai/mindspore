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

namespace mindspore {
namespace lite {
STATUS TfliteDeConvParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                                 const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                                 const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                 const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                                 schema::CNodeT *op, TensorCache *tensor_cache,  bool quantizedModel) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  MS_LOG(DEBUG) << "parse tflite Transpose_Conv parser";
  std::unique_ptr<schema::DeConv2DT> attr(new schema::DeConv2DT());
  const auto &tflite_attr = tfliteOp->builtin_options.AsTransposeConvOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: %s attr failed", op->name.c_str();
    return RET_NULL_PTR;
  }

  attr->group = 1;
  attr->strideW = tflite_attr->stride_w;
  attr->strideH = tflite_attr->stride_h;
  attr->dilateH = 1;
  attr->dilateW = 1;
  attr->padMode = GetPadMode(tflite_attr->padding);
  attr->format = schema::Format_NHWC;

  // get the conv op weight tensor
  auto weight_index = tfliteOp->inputs[1];
  const auto &weight_tensor = tfliteTensors[weight_index];
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "weight_tensor is null";
    return RET_NULL_PTR;
  }
  std::vector<tflite::TensorT *> weight_tensors{weight_tensor.get()};
  if (RET_OK != ParseTensor(weight_tensors, tfliteModelBuffer, tensor_cache, TF_CONST, true)) {
    return RET_ERROR;
  }
  auto weight_shape = weight_tensor->shape;
  attr->channelIn = weight_shape[CHWK_K];
  attr->channelOut = weight_shape[CHWK_C];
  attr->kernelW = weight_shape[CHWK_W];
  attr->kernelH = weight_shape[CHWK_H];

  op->primitive->value.type = schema::PrimitiveType_DeConv2D;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

TfliteNodeRegister g_tfliteDeConv2DParser("DeConv2D", new TfliteDeConvParser());
}  // namespace lite
}  // namespace mindspore

