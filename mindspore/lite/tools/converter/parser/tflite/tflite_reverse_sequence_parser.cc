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

#include "tools/converter/parser/tflite/tflite_reverse_sequence_parser.h"
#include <vector>
#include <memory>
#include "ops/reverse_sequence.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteReverseSequenceParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                    const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::ReverseSequence>();

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsReverseSequenceOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op reverse attr failed";
    return nullptr;
  }
  prim->set_seq_dim(tflite_attr->seq_dim);
  prim->set_batch_dim(tflite_attr->batch_dim);

  return prim.release();
}

TfliteNodeRegister g_tfliteReverseSequenceParser(tflite::BuiltinOperator_REVERSE_SEQUENCE,
                                                 new TfliteReverseSequenceParser());
}  // namespace lite
}  // namespace mindspore
