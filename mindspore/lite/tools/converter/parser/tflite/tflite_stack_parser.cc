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

#include "tools/converter/parser/tflite/tflite_stack_parser.h"
#include <vector>
#include <memory>
#include "ops/stack.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteStackParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Stack>();

  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  if (tflite_subgraph == nullptr) {
    MS_LOG(ERROR) << "tflite_subgraph is nullptr";
    return nullptr;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsPackOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op stack attr failed";
    return nullptr;
  }
  prim->set_axis(tflite_attr->axis);

  return prim.release();
}

TfliteNodeRegister g_tfliteStackParser(tflite::BuiltinOperator_PACK, new TfliteStackParser());
}  // namespace lite
}  // namespace mindspore
