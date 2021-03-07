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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_CUSTOM_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_CUSTOM_PARSER_H

#include <memory>
#include <vector>
#include <map>
#include "tools/converter/parser/tflite/tflite_node_parser.h"
#include "tools/converter/parser/tflite/tflite_node_parser_registry.h"

namespace mindspore {
namespace lite {
class TfliteCustomParser : public TfliteNodeParser {
 public:
  TfliteCustomParser() : TfliteNodeParser("Custom") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;

  static ops::PrimitiveC *DetectPostProcess(const std::vector<uint8_t> &custom_attr,
                                            const std::unique_ptr<tflite::OperatorT> &tflite_op);

  static ops::PrimitiveC *AudioSpectrogram(const std::vector<uint8_t> &custom_attr);

  static ops::PrimitiveC *Mfcc(const std::vector<uint8_t> &custom_attr);

  static ops::PrimitiveC *Predict(const std::vector<uint8_t> &custom_attr);

  static ops::PrimitiveC *Normalize();

  static ops::PrimitiveC *ExtractFeatures();

  ops::PrimitiveC *Rfft(const std::vector<uint8_t> &custom_attr, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                        const std::unique_ptr<tflite::ModelT> &tflite_model);

  static ops::PrimitiveC *FftReal();

  static ops::PrimitiveC *FftImag();

  static ops::PrimitiveC *Identity();
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_CUSTOM_PARSER_H
