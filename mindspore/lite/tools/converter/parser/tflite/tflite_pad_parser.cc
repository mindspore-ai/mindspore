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

#include "tools/converter/parser/tflite/tflite_pad_parser.h"
#include <vector>
#include <memory>
#include <string>
#include "ops/fusion/pad_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TflitePadParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::PadFusion>();

  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  if (tflite_subgraph == nullptr) {
    MS_LOG(ERROR) << "tflite_subgraph is nullptr";
    return nullptr;
  }
  auto &opcode = tflite_model->operator_codes.at(tflite_op->opcode_index);
  if (opcode == nullptr) {
    MS_LOG(ERROR) << "opcode is nullptr";
    return nullptr;
  }
  auto tflite_op_type = opcode->builtin_code;
  if (tflite_op_type == tflite::BuiltinOperator_PAD) {
    prim->set_padding_mode(mindspore::PaddingMode::CONSTANT);
    prim->set_constant_value(0.0);

    std::vector<std::vector<int64_t>> paddings;
    if (TransTfliteDataToVec2D(tflite_op->inputs.at(1), tflite_subgraph->tensors, tflite_model->buffers, paddings)) {
      MS_LOG(ERROR) << "get Pad -> paddings failed";
      return nullptr;
    }
    prim->set_paddings(paddings);
  } else if (tflite_op_type == tflite::BuiltinOperator_MIRROR_PAD) {
    const auto &tflite_attr = tflite_op->builtin_options.AsMirrorPadOptions();
    if (tflite_attr == nullptr) {
      MS_LOG(ERROR) << "get MirrorPad attr failed";
      return nullptr;
    }
    switch (tflite_attr->mode) {
      case tflite::MirrorPadMode_REFLECT:
        prim->set_padding_mode(mindspore::PaddingMode::REFLECT);
        break;
      case tflite::MirrorPadMode_SYMMETRIC:
        prim->set_padding_mode(mindspore::PaddingMode::SYMMETRIC);
        break;
      default:
        MS_LOG(ERROR) << "paddingMode:" << tflite_attr->mode << " is not supported";
        return nullptr;
    }
  } else {
    MS_LOG(ERROR) << "this pad:" << tflite_op_type << " hasn't been supported";
    return nullptr;
  }

  return prim.release();
}

TfliteNodeRegister g_tflitePadParser(tflite::BuiltinOperator_PAD, new TflitePadParser());
TfliteNodeRegister g_tfliteMirorPadParser(tflite::BuiltinOperator_MIRROR_PAD, new TflitePadParser());
}  // namespace lite
}  // namespace mindspore
