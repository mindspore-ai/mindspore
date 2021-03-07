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

#include "tools/converter/parser/tflite/tflite_topk_v2_parser.h"
#include <vector>
#include <memory>
#include "ops/fusion/topk_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteTopKV2Parser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                           const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::TopKFusion>();

  prim->set_sorted(true);

  return prim.release();
}

TfliteNodeRegister g_tfliteTopKV2Parser(tflite::BuiltinOperator_TOPK_V2, new TfliteTopKV2Parser());
}  // namespace lite
}  // namespace mindspore
