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

#include "tools/converter/parser/tflite/tflite_rank_parser.h"
#include <vector>
#include <memory>
#include "ops/rank.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteRankParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Rank>();
  return prim.release();
}

TfliteNodeRegister g_tfliteRankParser(tflite::BuiltinOperator_RANK, new TfliteRankParser());
}  // namespace lite
}  // namespace mindspore
