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

#include "tools/converter/parser/tflite/tflite_matmul_parser.h"
#include <memory>
#include "ops/mat_mul.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteMatMulParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                           const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                           const std::unique_ptr<tflite::ModelT> &tflite_model) {
  MS_CHECK_TRUE_RET(tflite_op != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_subgraph != nullptr, nullptr);
  auto prim = std::make_unique<ops::MatMul>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  const auto &tflite_attr = tflite_op->builtin_options.AsBatchMatMulOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op LRN attr failed";
    return nullptr;
  }
  prim->set_transpose_a(tflite_attr->adj_x);
  prim->set_transpose_b(tflite_attr->adj_y);

  return prim.release();
}
}  // namespace lite
}  // namespace mindspore
