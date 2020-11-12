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
#ifndef LITE_MODEL_PARSER_FOR_TFLITE_H
#define LITE_MODEL_PARSER_FOR_TFLITE_H

#include <string>
#include <unordered_map>
#include <memory>
#include "tools/converter/parser/tflite/tflite_model_parser.h"

namespace mindspore::lite {
class ModelParserForTflite : public TfliteModelParser {
 public:
  ModelParserForTflite() = default;

  ~ModelParserForTflite() override = default;

  FuncGraphPtr Parse(const std::string &modelFile, const std::string &weightFile, const QuantType &quantType) override;

 private:
  std::unordered_map<int, AnfNodePtr> nodes;
  std::unique_ptr<tflite::ModelT> tfliteModel;
  FuncGraphPtr funcGraphPtr;
  STATUS ConvertConstTensor(const tflite::TensorT *tensor, ParameterPtr parameter);
  STATUS ConvertOutputTensor(const tflite::OperatorT *op, CNodePtr dstCNode);
  STATUS ConvertOps();
  STATUS ConvertGraphInputs();
  STATUS ConvertGraphOutputs();
};
}  // namespace mindspore::lite
#endif  // LITE_MODEL_PARSER_FOR_TFLITE_H
