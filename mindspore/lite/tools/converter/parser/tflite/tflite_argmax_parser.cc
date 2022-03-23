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
PrimitiveCPtr TfliteArgmaxParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::ArgMaxFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_keep_dims(false);
  prim->set_out_max_value(false);
  prim->set_top_k(1);

  std::vector<int64_t> axes;
  auto ret = GetTfliteData(tflite_op->inputs[SECOND_INPUT], tflite_subgraph->tensors, tflite_model->buffers, &axes);
  if (ret != RET_OK && ret != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get axes value failed.";
    return nullptr;
  }
  if (axes.size() < 1) {
    MS_LOG(ERROR) << "invalid axes param";
    return nullptr;
  }
  prim->set_axis(axes.at(0));
  return prim->GetPrim();
}

TfliteNodeRegister g_tfliteArgmaxParser(tflite::BuiltinOperator_ARG_MAX, new TfliteArgmaxParser());
}  // namespace lite
}  // namespace mindspore
