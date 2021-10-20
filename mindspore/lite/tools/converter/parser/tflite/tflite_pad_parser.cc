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
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kTFlitePadInputSize = 3;
constexpr int kTFlitePaddingIndex = 1;
}  // namespace
ops::PrimitiveC *TflitePadParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  MS_CHECK_TRUE_RET(tflite_op != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_subgraph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_model != nullptr, nullptr);
  auto prim = std::make_unique<ops::PadFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  auto &opcode = tflite_model->operator_codes.at(tflite_op->opcode_index);
  if (opcode == nullptr) {
    MS_LOG(ERROR) << "opcode is nullptr";
    return nullptr;
  }
  auto tflite_op_type = opcode->builtin_code;
  if (tflite_op_type == tflite::BuiltinOperator_PAD) {
    MS_CHECK_GE(tflite_op->inputs.size(), kInputSize1, nullptr);
    prim->set_padding_mode(mindspore::PaddingMode::CONSTANT);
    prim->set_constant_value(0.0);

    std::vector<std::vector<int64_t>> paddings;
    if (TransTfliteDataToVec2D(tflite_op->inputs.at(SECOND_INPUT), tflite_subgraph->tensors, tflite_model->buffers,
                               &paddings)) {
      MS_LOG(ERROR) << "get Pad -> paddings failed";
      return nullptr;
    }
    prim->set_paddings(paddings);
  } else if (tflite_op_type == tflite::BuiltinOperator_PADV2) {
    MS_CHECK_GE(tflite_op->inputs.size(), kInputSize2, nullptr);
    prim->set_padding_mode(mindspore::PaddingMode::CONSTANT);
    if (tflite_op->inputs.size() < kTFlitePadInputSize) {
      MS_LOG(ERROR) << "tflite padv2 input size less than 3, which is " << tflite_op->inputs.size();
      return nullptr;
    }

    std::vector<float> constant_value;
    auto ret = GetTfliteData(tflite_op->inputs.at(THIRD_INPUT), tflite_subgraph->tensors, tflite_model->buffers,
                             &constant_value);
    if (ret != RET_OK || constant_value.size() != kTFlitePaddingIndex) {
      MS_LOG(ERROR) << "get Pad -> constant_value failed";
      return nullptr;
    }
    prim->set_constant_value(constant_value.at(0));
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
TfliteNodeRegister g_tflitePadV2Parser(tflite::BuiltinOperator_PADV2, new TflitePadParser());
TfliteNodeRegister g_tfliteMirorPadParser(tflite::BuiltinOperator_MIRROR_PAD, new TflitePadParser());
}  // namespace lite
}  // namespace mindspore
