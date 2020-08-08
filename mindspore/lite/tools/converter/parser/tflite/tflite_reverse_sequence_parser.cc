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
* distributed under the License is distributed on an AS
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <vector>
#include <memory>
#include "tools/converter/parser/tflite/tflite_reverse_sequence_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteReverseSequenceParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                                          const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                          const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tflite_opset,
                                          schema::CNodeT *op, TensorCache *tensor_cache, bool quantized_model) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  MS_LOG(DEBUG) << "parse TfliteReverseSequenceParser";
  std::unique_ptr<schema::ReverseSequenceT> attr(new schema::ReverseSequenceT());

  const auto &tflite_attr = tflite_op->builtin_options.AsReverseSequenceOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
    return RET_NULL_PTR;
  }
  attr->seqAxis = tflite_attr->seq_dim;
  attr->batchAxis = tflite_attr->batch_dim;

  if (GetTfliteData(tflite_op->inputs[1], tflite_tensors, tflite_model_buffer, attr->seqLengths)) {
    MS_LOG(ERROR) << "get reverse_sequence -> seqLengths failed";
    return RET_ERROR;
  }

  op->primitive->value.type = schema::PrimitiveType_ReverseSequence;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

TfliteNodeRegister g_tfliteReverseSequenceParser("ReverseSequence", new TfliteReverseSequenceParser());
}  // namespace lite
}  // namespace mindspore
