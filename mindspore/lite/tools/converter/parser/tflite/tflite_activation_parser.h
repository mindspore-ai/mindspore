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

#ifndef PREDICT_TFLITE_RELU_PARSER_H
#define PREDICT_TFLITE_RELU_PARSER_H

#include "tools/converter/parser/tflite/tflite_node_parser.h"
#include "tools/converter/parser/tflite/tflite_node_parser_registry.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {

class TfliteActivationParser : public TfliteNodeParser {
 public:
  TfliteActivationParser() : TfliteNodeParser("node_name") {}

  STATUS Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
               const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
               const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
               const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet, schema::CNodeT *op,
               TensorCache *tensor_cache, bool quantizedModel) override;
};

class TfliteReluParser : public TfliteActivationParser {
 public:
  TfliteReluParser() : TfliteActivationParser() {}
};

class TfliteRelu6Parser : public TfliteActivationParser{
 public:
  TfliteRelu6Parser() : TfliteActivationParser() {}
};

class TfliteTanhParser : public TfliteActivationParser{
 public:
  TfliteTanhParser() : TfliteActivationParser() {}
};

class TfliteLogisticParser : public TfliteActivationParser {
 public:
  TfliteLogisticParser() : TfliteActivationParser() {}
};

class TfliteLeakyReluParser : public TfliteActivationParser {
 public:
  TfliteLeakyReluParser() : TfliteActivationParser() {}
};

class TflitePreluParser : public TfliteNodeParser {
 public:
  TflitePreluParser() : TfliteNodeParser("Prelu") {}

  STATUS Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
               const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
               const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
               const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tflite_opset, schema::CNodeT *op,
               TensorCache *tensor_cache, bool quantized_model) override;
};

}  // namespace lite
}  // namespace mindspore

#endif  // PREDICT_TFLITE_RELU_PARSER_H

