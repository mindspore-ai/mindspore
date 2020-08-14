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

#include "tools/converter/parser/tflite/tflite_depthwise_conv_parser.h"
#include <vector>
#include <memory>
#include "tools/common/node_util.h"

namespace mindspore {
namespace lite {
STATUS TfliteDepthwiseConv2DParser::ParseGroupDepthwiseConv(schema::CNodeT *op,
                                                            const std::unique_ptr<schema::DepthwiseConv2DT> &attr,
                                                            const std::unique_ptr<tflite::TensorT> &weightTensor,
                                                            TensorCache *tensor_cache) {
  std::unique_ptr<schema::Conv2DT> convAttr(new schema::Conv2DT);

  convAttr->format = attr->format;
  convAttr->channelIn = attr->channelIn;
  convAttr->channelOut = attr->channelIn * attr->channelMultiplier;
  convAttr->kernelH = attr->kernelH;
  convAttr->kernelW = attr->kernelW;
  convAttr->strideH = attr->strideH;
  convAttr->strideW = attr->strideW;
  convAttr->padMode = attr->padMode;
  convAttr->padUp = attr->padUp;
  convAttr->padDown = attr->padDown;
  convAttr->padLeft = attr->padLeft;
  convAttr->padRight = attr->padRight;
  convAttr->dilateH = attr->dilateH;
  convAttr->dilateW = attr->dilateW;
  convAttr->hasBias = attr->hasBias;
  convAttr->activationType = attr->activationType;

  auto weightTensorIndex = tensor_cache->FindTensor(weightTensor->name);
  if (weightTensorIndex >= 0 && weightTensorIndex < tensor_cache->GetCachedTensor().size()) {
    auto liteWeightTensor = tensor_cache->GetCachedTensor()[weightTensorIndex];
    if (liteWeightTensor->dataType == TypeId::kNumberTypeUInt8) {
      // convert weight format KHWC -> CHWK
      auto status = TransFilterFormat<uint8_t>(liteWeightTensor, kKHWC2CHWK);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Trans depthwiseConv Filter Format failed.";
        return RET_ERROR;
      }
    }

    if (liteWeightTensor->dataType == kNumberTypeFloat32 || liteWeightTensor->dataType == kNumberTypeFloat) {
      // convert weight format KHWC -> CHWK
      auto status = TransFilterFormat<float>(liteWeightTensor, kKHWC2CHWK);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Trans depthwiseConv Filter Format failed.";
        return RET_ERROR;
      }
    }
  }

  op->primitive->value.type = schema::PrimitiveType_Conv2D;
  op->primitive->value.value = convAttr.release();
  return RET_OK;
}

STATUS TfliteDepthwiseConv2DParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
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

  MS_LOG(DEBUG) << "parse TfliteDepthwiseConv2DParser";
  std::unique_ptr<schema::DepthwiseConv2DT> attr(new schema::DepthwiseConv2DT());
  const auto &tflite_attr = tflite_op->builtin_options.AsDepthwiseConv2DOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
    return RET_NULL_PTR;
  }
  attr->strideW = tflite_attr->stride_w;
  attr->strideH = tflite_attr->stride_h;
  attr->dilateH = tflite_attr->dilation_h_factor;
  attr->dilateW = tflite_attr->dilation_w_factor;
  attr->padMode = GetPadMode(tflite_attr->padding);
  attr->format = schema::Format_NHWC;
  attr->activationType = GetActivationFunctionType(tflite_attr->fused_activation_function);
  // get the conv op weight tensor
  auto input_index = tflite_op->inputs[0];
  const auto &input_tenosr = tflite_tensors[input_index];
  if (input_tenosr == nullptr) {
    MS_LOG(ERROR) << "the first input is null";
    return RET_NULL_PTR;
  }
  auto input_shape = input_tenosr->shape;

  auto weight_index = tflite_op->inputs[1];
  const auto &weight_tensor = tflite_tensors[weight_index];
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "the weight tensor is null";
    return RET_NULL_PTR;
  }
  auto weight_shape = weight_tensor->shape;
  attr->channelIn = input_shape[KHWC_C];
  attr->channelMultiplier = tflite_attr->depth_multiplier;
  attr->kernelH = weight_shape[KHWC_H];
  attr->kernelW = weight_shape[KHWC_W];

  std::vector<tflite::TensorT *> weight_tensors{weight_tensor.get()};

  if (RET_OK != ParseTensor(weight_tensors, tfliteModelBuffer, tensor_cache, TF_CONST, true)) {
    MS_LOG(ERROR) << "parse weight failed";
    return RET_ERROR;
  }

  if (tflite_op->inputs.size() == 3) {
    attr->hasBias = true;
    auto bias_index = tflite_op->inputs[2];
    const auto &bias_tensor = tflite_tensors[bias_index];
    std::vector<tflite::TensorT *> bias_tensors{bias_tensor.get()};
    if (RET_OK != ParseTensor(bias_tensors, tfliteModelBuffer, tensor_cache, TF_CONST, false)) {
      MS_LOG(ERROR) << "parse bias failed";
      return RET_ERROR;
    }
  }

  if (attr->channelMultiplier > 1) {
    if (RET_OK != ParseGroupDepthwiseConv(op, attr, weight_tensor, tensor_cache)) {
      MS_LOG(ERROR) << "Parse Group DepthwiseConv failed";
      return RET_ERROR;
    }
  } else {
    op->primitive->value.type = schema::PrimitiveType_DepthwiseConv2D;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

TfliteNodeRegister g_tfliteDepthwiseConv2DParser("DepthwiseConv2D", new TfliteDepthwiseConv2DParser());
}  // namespace lite
}  // namespace mindspore


