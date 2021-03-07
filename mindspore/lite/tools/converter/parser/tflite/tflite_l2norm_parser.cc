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

#include "tools/converter/parser/tflite/tflite_l2norm_parser.h"
#include <vector>
#include <memory>
#include "ops/fusion/l2_normalize_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteL2NormParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                           const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::L2NormalizeFusion>();

  prim->set_axis({-1});
  prim->set_epsilon(1e-6f);

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsL2NormOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get L2NormalizeFusion attr failed";
    return nullptr;
  }
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  return prim.release();
}

TfliteNodeRegister g_tfliteL2NormParser(tflite::BuiltinOperator_L2_NORMALIZATION, new TfliteL2NormParser());
}  // namespace lite
}  // namespace mindspore
