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

#include "node_parser/add_parser_tutorial.h"
#include <memory>
#include "include/registry/node_parser_registry.h"
#include "ops/fusion/add_fusion.h"

namespace mindspore {
namespace converter {
ops::BaseOperatorPtr AddParserTutorial::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                              const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                              const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = api::MakeShared<ops::AddFusion>();
  if (prim == nullptr) {
    return nullptr;
  }
  prim->set_activation_type(mindspore::NO_ACTIVATION);  // user need to analyze tflite_op's attr.
  return prim;
}

REG_NODE_PARSER(kFmkTypeTflite, ADD, std::make_shared<AddParserTutorial>());
}  // namespace converter
}  // namespace mindspore
