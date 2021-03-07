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

#include "tools/converter/parser/tflite/tflite_argmin_parser.h"
#include <memory>
#include <vector>
#include <map>
#include "ops/fusion/arg_min_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteArgminParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                           const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::ArgMinFusion>();

  prim->set_keep_dims(false);
  prim->set_out_max_value(false);
  prim->set_top_k(1);

  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  if (tflite_subgraph == nullptr) {
    MS_LOG(ERROR) << "tflite_subgraph is nullptr";
    return nullptr;
  }
  const auto &axis_tensor = tflite_subgraph->tensors.at(tflite_op->inputs[1]);
  if (axis_tensor == nullptr) {
    MS_LOG(ERROR) << "axis_tensor is nullptr";
    return nullptr;
  }
  const auto &buf_data = tflite_model->buffers.at(axis_tensor->buffer);
  if (buf_data == nullptr) {
    MS_LOG(ERROR) << "the buf data is null";
    return nullptr;
  }
  auto data_ptr = buf_data->data.data();
  if (data_ptr == nullptr) {
    MS_LOG(ERROR) << "the data is null";
    return nullptr;
  }
  prim->set_axis(*(static_cast<int64_t *>(static_cast<void *>(data_ptr))));

  return prim.release();
}

TfliteNodeRegister g_tfliteArgminParser(tflite::BuiltinOperator_ARG_MIN, new TfliteArgminParser());
}  // namespace lite
}  // namespace mindspore
