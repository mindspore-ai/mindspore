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

#include "tools/converter/parser/tflite/tflite_range_parser.h"
#include <vector>
#include <memory>
#include "ops/range.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteRangeParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  MS_CHECK_TRUE_RET(tflite_op != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_subgraph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_model != nullptr, nullptr);
  MS_CHECK_GE(tflite_op->inputs.size(), kInputSize2, nullptr);
  auto prim = std::make_unique<ops::Range>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  prim->set_d_type(0);

  std::vector<int64_t> limit;
  std::vector<int64_t> delta;
  int ret = GetTfliteData(tflite_op->inputs.at(SECOND_INPUT), tflite_subgraph->tensors, tflite_model->buffers, &limit);
  if (ret != RET_OK && ret != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get range -> limit failed";
    return nullptr;
  }
  if (ret == RET_OK) {
    ret = GetTfliteData(tflite_op->inputs.at(THIRD_INPUT), tflite_subgraph->tensors, tflite_model->buffers, &delta);
    if (ret != RET_OK && ret != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "get range -> delta failed";
      return nullptr;
    }
  }
  if (ret == RET_OK) {
    prim->set_limit(limit.front());
    prim->set_delta(delta.front());
  }

  return prim.release();
}

TfliteNodeRegister g_tfliteRangeParser(tflite::BuiltinOperator_RANGE, new TfliteRangeParser());
}  // namespace lite
}  // namespace mindspore
