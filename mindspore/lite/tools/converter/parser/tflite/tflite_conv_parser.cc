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

namespace mindspore {
namespace lite {
STATUS TfliteConvParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                               const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                               const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                               const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                               schema::CNodeT *op, TensorCache *tensor_cache, bool quantizedModel) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  MS_LOG(DEBUG) << "parse TfliteConvParser";
  std::unique_ptr<schema::Conv2DT> attr(new schema::Conv2DT());
  const auto &tfliteAttr = tfliteOp->builtin_options.AsConv2DOptions();
  if (tfliteAttr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
    return RET_NULL_PTR;
  }
  attr->group = 1;
  attr->strideW = tfliteAttr->stride_w;
  attr->strideH = tfliteAttr->stride_h;
  attr->dilateH = tfliteAttr->dilation_h_factor;
  attr->dilateW = tfliteAttr->dilation_w_factor;
  attr->padMode = GetPadMode(tfliteAttr->padding);
  attr->format = schema::Format_NHWC;
  attr->activationType = GetActivationFunctionType(tfliteAttr->fused_activation_function);

  // get the conv op weight tensor
  auto weight_index = tfliteOp->inputs[1];
  const auto &weight_tensor = tfliteTensors[weight_index];
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "weight_tensor is null";
    return RET_NULL_PTR;
  }
  std::vector<tflite::TensorT *> weight_tensors{weight_tensor.get()};
  if (RET_OK != ParseTensor(weight_tensors, tfliteModelBuffer, tensor_cache, TF_CONST, true)) {
    MS_LOG(ERROR) << "parse weight failed";
    return RET_ERROR;
  }
  auto weight_shape = weight_tensor->shape;
  attr->channelIn = weight_shape[KHWC_C];
  attr->channelOut = weight_shape[KHWC_K];
  attr->kernelW = weight_shape[KHWC_W];
  attr->kernelH = weight_shape[KHWC_H];

  // get the conv op bias tensor
  if (tfliteOp->inputs.size() == 3) {
    attr->hasBias = true;
    auto bias_index = tfliteOp->inputs[2];
    const auto &bias_tensor = tfliteTensors[bias_index];
    if (bias_tensor == nullptr) {
      MS_LOG(ERROR) << "bias_tensor is null";
      return RET_NULL_PTR;
    }
    std::vector<tflite::TensorT *> bias_tensors{bias_tensor.get()};
    if (RET_OK != ParseTensor(bias_tensors, tfliteModelBuffer, tensor_cache, TF_CONST, false)) {
      MS_LOG(ERROR) << "parse bias failed";
      return RET_ERROR;
    }
  }

  // calculate pad params

  op->primitive->value.type = schema::PrimitiveType_Conv2D;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

TfliteNodeRegister g_tfliteConv2DParser("Conv2D", new TfliteConvParser());
}  // namespace lite
}  // namespace mindspore


