/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_EXAMPLES_CONVERTER_EXTEND_NODE_PARSER_ADD_PARSER_TUTORIAL_H
#define MINDSPORE_LITE_EXAMPLES_CONVERTER_EXTEND_NODE_PARSER_ADD_PARSER_TUTORIAL_H

#include <memory>
#include "include/registry/node_parser.h"

namespace mindspore {
namespace converter {
class AddParserTutorial : public NodeParser {
 public:
  AddParserTutorial() = default;
  ~AddParserTutorial() = default;
  ops::BaseOperatorPtr Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                             const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                             const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};
}  // namespace converter
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXAMPLES_CONVERTER_EXTEND_NODE_PARSER_ADD_PARSER_TUTORIAL_H
