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
#include <map>
#include "tools/common/node_util.h"

namespace mindspore {
namespace lite {
STATUS TfliteDequantizeParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                                     const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                     schema::CNodeT *op,
                                     std::vector<int32_t> *tensors_id,
                                     std::vector<schema::Format> *tensors_format,
                                     std::map<int, int>  *tensors_id_map) {
  MS_LOG(DEBUG) << "parse TfliteDequantizeNParser";

  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::CastT> attr(new schema::CastT);

  // get the dequantize input tensor
  const auto &in_tensor = tflite_tensors[tflite_op->inputs[0]];
  if (in_tensor == nullptr) {
    MS_LOG(ERROR) << "input tensor is null";
    return RET_NULL_PTR;
  }
  attr->srcT = GetTfliteDataType(in_tensor->type);
  const auto &out_tensor = tflite_tensors[tflite_op->outputs[0]];
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "output tensor is null";
    return RET_NULL_PTR;
  }
  attr->dstT = GetTfliteDataType(out_tensor->type);

  op->primitive->value.type = schema::PrimitiveType_Fp16Cast;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_id, tensors_format, tensors_id_map,
             tflite_op->inputs[0], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  AddOpOutput(op, tensors_id, tensors_format, tensors_id_map,
              tflite_op->outputs[0], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteDequantizeParser("DEQUANTIZE", new TfliteDequantizeParser());
}  // namespace lite
}  // namespace mindspore
