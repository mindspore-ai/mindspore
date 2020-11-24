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

#include "tools/converter/parser/tflite/tflite_strided_slice_parser.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {
PrimitiveC *TfliteStridedSliceParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto &tflite_subgraph = tflite_model->subgraphs.front();
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is null";
    return nullptr;
  }

  std::unique_ptr<schema::StridedSliceT> attr = std::make_unique<schema::StridedSliceT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsStridedSliceOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op strideslice attr failed";
    return nullptr;
  }
  attr->beginMask = tflite_attr->begin_mask;
  attr->endMask = tflite_attr->end_mask;
  attr->ellipsisMask = tflite_attr->ellipsis_mask;
  attr->newAxisMask = tflite_attr->new_axis_mask;
  attr->shrinkAxisMask = tflite_attr->shrink_axis_mask;

  int status = GetTfliteData(tflite_op->inputs[1], tflite_subgraph->tensors, tflite_model->buffers, attr->begin);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "stridedSlice -> begin get failed";
    return nullptr;
  } else if (status == RET_OK) {
    status = GetTfliteData(tflite_op->inputs[2], tflite_subgraph->tensors, tflite_model->buffers, attr->end);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "stridedSlice -> end get failed";
      return nullptr;
    } else if (status == RET_OK) {
      status = GetTfliteData(tflite_op->inputs[3], tflite_subgraph->tensors, tflite_model->buffers, attr->stride);
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "stridedSlice -> stride get failed";
        return nullptr;
      }
    }
  }
  attr->isScale.assign(tflite_subgraph->tensors[tflite_op->inputs[0]]->shape.begin(),
                       tflite_subgraph->tensors[tflite_op->inputs[0]]->shape.end());

  primitive->value.type = schema::PrimitiveType_StridedSlice;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteStridedSliceParser(tflite::BuiltinOperator_STRIDED_SLICE, new TfliteStridedSliceParser());
}  // namespace lite
}  // namespace mindspore
