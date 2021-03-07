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
#include "ops/strided_slice.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteStridedSliceParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                 const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::StridedSlice>();

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsStridedSliceOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get strideslice attr failed";
    return nullptr;
  }
  prim->set_begin_mask(tflite_attr->begin_mask);
  prim->set_end_mask(tflite_attr->end_mask);
  prim->set_ellipsis_mask(tflite_attr->ellipsis_mask);
  prim->set_new_axis_mask(tflite_attr->new_axis_mask);
  prim->set_shrink_axis_mask(tflite_attr->shrink_axis_mask);

  return prim.release();
}

TfliteNodeRegister g_tfliteStridedSliceParser(tflite::BuiltinOperator_STRIDED_SLICE, new TfliteStridedSliceParser());
}  // namespace lite
}  // namespace mindspore
