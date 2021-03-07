/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/tflite/tflite_dequantize_parser.h"
#include <vector>
#include <memory>
#include "ops/quant_dtype_cast.h"
#include "ops/cast.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteDequantizeParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                               const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto &tflite_subgraph = tflite_model->subgraphs.front();
  if (tflite_subgraph == nullptr) {
    MS_LOG(ERROR) << "tflite_subgraph is nullptr";
    return nullptr;
  }
  const auto &in_tensor = tflite_subgraph->tensors[tflite_op->inputs.at(0)];
  if (in_tensor == nullptr) {
    MS_LOG(ERROR) << "input tensor is null";
    return nullptr;
  }
  const auto &out_tensor = tflite_subgraph->tensors[tflite_op->outputs.at(0)];
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "output tensor is null";
    return nullptr;
  }
  if ((GetTfliteDataType(in_tensor->type) == kNumberTypeInt8 ||
       GetTfliteDataType(in_tensor->type) == kNumberTypeUInt8)) {
    auto prim = std::make_unique<ops::QuantDTypeCast>();
    prim->set_src_t(GetTfliteDataType(in_tensor->type));
    prim->set_dst_t(GetTfliteDataType(out_tensor->type));
    return prim.release();
  } else {
    auto prim = std::make_unique<ops::Cast>();
    auto dstT = GetTfliteDataType(out_tensor->type);
    prim->AddAttr("to", MakeValue(static_cast<int32_t>(dstT)));
    return prim.release();
  }
}

TfliteNodeRegister g_tfliteDequantizeParser(tflite::BuiltinOperator_DEQUANTIZE, new TfliteDequantizeParser());
}  // namespace lite
}  // namespace mindspore
