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

#include "tools/converter/parser/tflite/tflite_gather_parser.h"
#include <vector>
#include <memory>
#include "ops/gather.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TfliteGatherParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Gather>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);

  const auto &tflite_attr = tflite_op->builtin_options.AsGatherOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op gather attr failed";
    return nullptr;
  }
  auto axis_value = MakeValue(static_cast<int32_t>(tflite_attr->axis));
  MS_CHECK_TRUE_RET(axis_value != nullptr, nullptr);
  (void)prim_c->AddAttr("axis", axis_value);

  return prim->GetPrim();
}

TfliteNodeRegister g_tfliteGatherParser(tflite::BuiltinOperator_GATHER, new TfliteGatherParser());
}  // namespace lite
}  // namespace mindspore
