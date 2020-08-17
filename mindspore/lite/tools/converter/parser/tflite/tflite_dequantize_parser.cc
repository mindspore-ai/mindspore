/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/tflite/tflite_dequantize_parser.h"
#include <vector>
#include <memory>
#include "tools/common/node_util.h"

namespace mindspore {
namespace lite {
STATUS TfliteDequantizeParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
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

  MS_LOG(DEBUG) << "parse TfliteDequantizeNParser";
  std::unique_ptr<schema::CastT> attr(new schema::CastT);

  // get the dequantize input tensor
  const auto &in_tensor = tfliteTensors[tfliteOp->inputs[0]];
  if (in_tensor == nullptr) {
    MS_LOG(ERROR) << "weight_tensor is null";
    return RET_NULL_PTR;
  }
  attr->srcT = dtype_map[in_tensor->type];

  const auto &out_tensor = tfliteTensors[tfliteOp->outputs[0]];
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "tensor is null";
    return RET_NULL_PTR;
  }
  attr->dstT = dtype_map[out_tensor->type];
  std::vector<tflite::TensorT *> weight_tensors{in_tensor.get()};
  if (RET_OK != ParseTensor(weight_tensors, tfliteModelBuffer, tensor_cache, TF_CONST, true)) {
    MS_LOG(ERROR) << "parse weight failed";
    return RET_ERROR;
  }

  op->primitive->value.type = schema::PrimitiveType_Cast;
  op->primitive->value.value = attr.release();
  return 0;
}

TfliteNodeRegister g_tfliteDequantizeParser("DEQUANTIZE", new TfliteDequantizeParser());
}  // namespace lite
}  // namespace mindspore
