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

#include "tools/converter/parser/tflite/tflite_argmax_parser.h"
#include <memory>
#include <vector>
#include <map>
#include "ops/fusion/arg_max_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteArgmaxParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                           const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                           const std::unique_ptr<tflite::ModelT> &tflite_model) {
  MS_CHECK_TRUE_RET(tflite_op != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_subgraph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_model != nullptr, nullptr);
  auto prim = std::make_unique<ops::ArgMaxFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_keep_dims(false);
  prim->set_out_max_value(false);
  prim->set_top_k(1);

  const auto &axis_tensor = tflite_subgraph->tensors.at(tflite_op->inputs[1]);
  MS_CHECK_TRUE_MSG(axis_tensor != nullptr, nullptr, "axis_tensor is nullptr");
  const auto &buf_data = tflite_model->buffers.at(axis_tensor->buffer);
  MS_CHECK_TRUE_MSG(buf_data != nullptr, nullptr, "the buf data is nullptr");
  auto data_ptr = buf_data->data.data();
  if (data_ptr == nullptr) {
    MS_LOG(ERROR) << "the data is null";
    return nullptr;
  }
  prim->set_axis(*(static_cast<int64_t *>(static_cast<void *>(data_ptr))));

  return prim.release();
}

TfliteNodeRegister g_tfliteArgmaxParser(tflite::BuiltinOperator_ARG_MAX, new TfliteArgmaxParser());
}  // namespace lite
}  // namespace mindspore
