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

#include "tools/converter/parser/tflite/tflite_reshape_parser.h"
#include <vector>
#include <memory>
#include "ops/reshape.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TfliteReshapeParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                         const std::unique_ptr<tflite::ModelT> &tfliteModel) {
  auto prim = std::make_unique<ops::Reshape>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  std::vector<int32_t> shape;
  const auto &tflite_attr = tflite_op->builtin_options.AsReshapeOptions();
  if (tflite_attr != nullptr) {
    shape.resize(tflite_attr->new_shape.size());
    for (size_t i = 0; i < tflite_attr->new_shape.size(); ++i) {
      shape[i] = tflite_attr->new_shape[i];
    }
    auto value_ptr = MakeValue(shape);
    MS_CHECK_TRUE_RET(value_ptr != nullptr, nullptr);
    (void)prim_c->AddAttr("shape", value_ptr);
  }

  return prim->GetPrim();
}

TfliteNodeRegister g_tfliteReshapeParser(tflite::BuiltinOperator_RESHAPE, new TfliteReshapeParser());
}  // namespace lite
}  // namespace mindspore
