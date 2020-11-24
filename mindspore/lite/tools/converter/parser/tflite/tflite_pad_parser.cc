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

namespace mindspore {
namespace lite {
PrimitiveC *TflitePadParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto &tflite_subgraph = tflite_model->subgraphs.front();
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is null";
    return nullptr;
  }

  std::unique_ptr<schema::PadT> attr = std::make_unique<schema::PadT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto tflite_op_type = (tflite_model->operator_codes[tflite_op->opcode_index])->builtin_code;
  if (tflite_op_type == tflite::BuiltinOperator_PAD) {
    const auto &tflite_attr = tflite_op->builtin_options.AsPadOptions();
    if (tflite_attr == nullptr) {
      MS_LOG(ERROR) << "get op pad attr failed";
      return nullptr;
    }
    attr->paddingMode = schema::PaddingMode_CONSTANT;
    attr->constantValue = 0.0f;
    if (GetTfliteData(tflite_op->inputs[1], tflite_subgraph->tensors, tflite_model->buffers, attr->paddings)) {
      MS_LOG(ERROR) << "get pad -> paddings failed";
      return nullptr;
    }
  } else if (tflite_op_type == tflite::BuiltinOperator_MIRROR_PAD) {
    const auto &tflite_attr = tflite_op->builtin_options.AsMirrorPadOptions();
    if (tflite_attr == nullptr) {
      MS_LOG(ERROR) << "get op pad attr failed";
      return nullptr;
    }
    switch (tflite_attr->mode) {
      case tflite::MirrorPadMode_REFLECT:
        attr->paddingMode = schema::PaddingMode_REFLECT;
        break;
      case tflite::MirrorPadMode_SYMMETRIC:
        attr->paddingMode = schema::PaddingMode_SYMMETRIC;
        break;
      default:
        MS_LOG(ERROR) << "paddingmode:" << tflite_attr->mode << " don't support";
        return nullptr;
    }
  } else {
    MS_LOG(ERROR) << "this pad:" << tflite_op_type << " hasn't been supported";
    return nullptr;
  }

  primitive->value.type = schema::PrimitiveType_Pad;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tflitePadParser(tflite::BuiltinOperator_PAD, new TflitePadParser());
TfliteNodeRegister g_tfliteMirorPadParser(tflite::BuiltinOperator_MIRROR_PAD, new TflitePadParser());
}  // namespace lite
}  // namespace mindspore
