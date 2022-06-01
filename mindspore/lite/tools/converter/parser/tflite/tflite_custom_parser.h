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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_CUSTOM_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_CUSTOM_PARSER_H_

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

  ~TfliteCustomParser() override = default;

  PrimitiveCPtr Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                      const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                      const std::unique_ptr<tflite::ModelT> &tflite_model) override;

  static PrimitiveCPtr DetectPostProcess(const std::vector<uint8_t> &custom_attr,
                                         const std::unique_ptr<tflite::OperatorT> &tflite_op);

  static PrimitiveCPtr AudioSpectrogram(const std::vector<uint8_t> &custom_attr);

  static PrimitiveCPtr Mfcc(const std::vector<uint8_t> &custom_attr);

  static PrimitiveCPtr Predict(const std::vector<uint8_t> &custom_attr);

  static PrimitiveCPtr Normalize();

  static PrimitiveCPtr ExtractFeatures();

  PrimitiveCPtr Rfft(const std::vector<uint8_t> &custom_attr, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                     const std::unique_ptr<tflite::ModelT> &tflite_model);

  static PrimitiveCPtr FftReal();

  static PrimitiveCPtr FftImag();

  static PrimitiveCPtr Identity();
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_CUSTOM_PARSER_H_
