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
#include <utility>
#include "tools/converter/parser/tflite/tflite_scatter_nd_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteScatterNdParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                                    const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                                    const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                    const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                                    schema::CNodeT *op, TensorCache *tensor_cache, bool quantizedModel) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  MS_LOG(INFO) << "parse TfliteScatterNdParser";
  std::unique_ptr<schema::ScatterNDT> attr(new schema::ScatterNDT());

  const auto &tflite_attr = tfliteOp->builtin_options.AsScatterNdOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name << " attr failed";
    return RET_NULL_PTR;
  }
  /*
  MS_LOG(DEBUG) << "op->inputIndex";
  for (auto &i : op->inputIndex) {
    MS_LOG(DEBUG) << i;
  }
   */
  // in tflite, kIndices = 0, kUpdates = 1, kShape = 2
  // in mslite, kScatterShapeIndex = 0, kScatterIndicesIndex = 1, kScatterUpdateIndex = 2;
  std::swap(op->inputIndex[0], op->inputIndex[2]);
  std::swap(op->inputIndex[1], op->inputIndex[2]);
  /*
  MS_LOG(DEBUG) << "op->inputIndex after resort";
  for (auto &i : op->inputIndex) {
    MS_LOG(DEBUG) << i;
  }
   */

  op->primitive->value.type = schema::PrimitiveType_ScatterND;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

TfliteNodeRegister g_TfliteScatterNdParser("ScatterNd", new TfliteScatterNdParser());
}  // namespace lite
}  // namespace mindspore
