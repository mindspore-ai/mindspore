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

#include "tools/converter/parser/tflite/tflite_slice_parser.h"
#include <vector>
#include <memory>
#include "ops/fusion/slice_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteSliceParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  MS_CHECK_TRUE_RET(tflite_op != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_subgraph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_model != nullptr, nullptr);
  MS_CHECK_GE(tflite_op->inputs.size(), kInputSize1, nullptr);
  auto prim = std::make_unique<ops::SliceFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  std::vector<int64_t> begin;
  auto ret = GetTfliteData(tflite_op->inputs[1], tflite_subgraph->tensors, tflite_model->buffers, &begin);
  if (ret != RET_OK && ret != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get slice -> begin failed";
    return nullptr;
  }
  std::vector<int64_t> axes;
  for (size_t i = 0; i < begin.size(); ++i) {
    axes.push_back(i);
  }
  prim->set_axes(axes);

  return prim.release();
}

TfliteNodeRegister g_tfliteSliceParser(tflite::BuiltinOperator_SLICE, new TfliteSliceParser());
}  // namespace lite
}  // namespace mindspore
