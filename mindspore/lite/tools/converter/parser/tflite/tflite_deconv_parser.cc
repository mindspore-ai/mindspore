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
#include "tools/converter/parser/tflite/tflite_deconv_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteDeConvParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                               const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                               const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                               const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tflite_op_set,
                               schema::CNodeT *op,
                               TensorCache *tensor_cache, bool quantized_model) {
  MS_LOG(DEBUG) << "parse tflite Transpose_Conv parser";
  std::unique_ptr<schema::DeConv2DT> attr(new schema::DeConv2DT());
  const auto &tflite_attr = tflite_op->builtin_options.AsTransposeConvOptions();
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
  auto weight_index = tflite_op->inputs[1];
  const auto &weight_tensor = tflite_tensors[weight_index];
  std::vector<tflite::TensorT *> weight_tensors{weight_tensor.get()};

  if (RET_OK != ParseWeight(weight_tensors, tflite_model_buffer, tensor_cache, schema::Format_KHWC)) {
    return RET_ERROR;
  }
  auto weight_shape = weight_tensor->shape;
  attr->channelIn = weight_shape[KHWC_C];
  attr->channelOut = weight_shape[KHWC_K];
  attr->kernelW = weight_shape[KHWC_W];
  attr->kernelH = weight_shape[KHWC_H];

  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_DeConv2D;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

TfliteNodeRegister g_tfliteDeConv2DParser("DeConv2D", new TfliteDeConvParser());
}  // namespace lite
}  // namespace mindspore

