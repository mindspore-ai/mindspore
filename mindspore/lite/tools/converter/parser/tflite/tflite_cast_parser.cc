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
#include "tools/converter/parser/tflite/tflite_cast_parser.h"
#include <vector>
#include <memory>
#include "ops/cast.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TfliteCastParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                      const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                      const std::unique_ptr<tflite::ModelT> &tflite_model) {
  MS_CHECK_TRUE_RET(!tflite_op->outputs.empty(), nullptr);
  auto prim = std::make_unique<ops::Cast>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);

  const auto &out_tensor = tflite_subgraph->tensors[tflite_op->outputs.front()];
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "tensor is null";
    return nullptr;
  }
  auto dstT = GetTfliteDataType(out_tensor->type);
  auto value_dst = MakeValue(static_cast<int32_t>(dstT));
  MS_CHECK_TRUE_RET(value_dst != nullptr, nullptr);
  (void)prim_c->AddAttr("to", value_dst);

  return prim->GetPrim();
}

TfliteNodeRegister g_tfliteCastParser(tflite::BuiltinOperator_CAST, new TfliteCastParser());
}  // namespace lite
}  // namespace mindspore
