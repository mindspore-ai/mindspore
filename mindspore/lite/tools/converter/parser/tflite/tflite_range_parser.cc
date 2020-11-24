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

namespace mindspore {
namespace lite {
PrimitiveC *TfliteRangeParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                  const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto &tflite_subgraph = tflite_model->subgraphs.front();
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is null";
    return nullptr;
  }

  std::unique_ptr<schema::RangeT> attr = std::make_unique<schema::RangeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  attr->dType = 0;
  std::vector<int> limit;
  std::vector<int> delta;
  int status = GetTfliteData(tflite_op->inputs[1], tflite_subgraph->tensors, tflite_model->buffers, limit);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "range -> limit get failed";
    return nullptr;
  } else if (status == RET_OK) {
    status = GetTfliteData(tflite_op->inputs[2], tflite_subgraph->tensors, tflite_model->buffers, delta);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "stridedSlice -> end get failed";
      return nullptr;
    }
  }
  if (status == RET_OK) {
    attr->limit = limit.front();
    attr->delta = delta.front();
  }
  primitive->value.type = schema::PrimitiveType_Range;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteRangeParser(tflite::BuiltinOperator_RANGE, new TfliteRangeParser());
}  // namespace lite
}  // namespace mindspore
