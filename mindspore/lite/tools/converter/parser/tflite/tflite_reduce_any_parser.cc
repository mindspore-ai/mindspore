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

#include <vector>
#include <memory>
#include "tools/converter/parser/tflite/tflite_reduce_any_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteReduceAnyParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                                    const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                                    const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                    const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                                    schema::CNodeT *op, TensorCache *tensor_cache, bool quantizedModel) {
  MS_LOG(INFO) << "parse TfliteReduceAnyParser";
  std::unique_ptr<schema::ReduceT> attr(new schema::ReduceT());
  const auto &tflite_attr = tfliteOp->builtin_options.AsReducerOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name << " attr failed";
    return RET_NULL_PTR;
  }

  attr->keepDims = tflite_attr->keep_dims;
  // attr->mode = schema::;
  MS_LOG(ERROR) << "ms-lite haven't supported REDUCE_ANY now";
  return RET_NOT_FIND_OP;

  if (GetTfliteData(tfliteOp->inputs[1], tfliteTensors, tfliteModelBuffer, attr->axes)) {
    MS_LOG(ERROR) << "get reduce_any->axes failed";
    return RET_ERROR;
  }

  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Reduce;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

TfliteNodeRegister g_TfliteReduceAnyParser("ReduceAny", new TfliteReduceAnyParser());
}  // namespace lite
}  // namespace mindspore
