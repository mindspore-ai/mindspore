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
#include <map>
#include <string>

namespace mindspore {
namespace lite {
STATUS TflitePadParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                              const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                              const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                              schema::CNodeT *op, std::vector<int32_t> *tensors_id,
                              std::vector<schema::Format> *tensors_format, std::map<int, int> *tensors_id_map) {
  MS_LOG(DEBUG) << "parse TflitePadParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::PadT> attr = std::make_unique<schema::PadT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  std::vector<std::string> node_name_str;
  Split(op->name, &node_name_str, "-");
  const char *node_name = node_name_str.data()->c_str();
  if (std::strcmp(node_name, "Pad") == 0) {
    const auto &tflite_attr = tflite_op->builtin_options.AsPadOptions();
    if (tflite_attr == nullptr) {
      MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
      return RET_NULL_PTR;
    }
    attr->paddingMode = schema::PaddingMode_CONSTANT;
    attr->constantValue = 0.0f;
    if (GetTfliteData(tflite_op->inputs[1], tflite_tensors, tflite_model_buffer, attr->paddings)) {
      MS_LOG(ERROR) << "get pad -> paddings failed";
      return RET_ERROR;
    }
  } else if (std::strcmp(node_name, "MirrorPad") == 0) {
    const auto &tflite_attr = tflite_op->builtin_options.AsMirrorPadOptions();
    if (tflite_attr == nullptr) {
      MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
      return RET_NULL_PTR;
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
        return RET_INVALID_OP_ATTR;
      }
  } else {
    MS_LOG(ERROR) << "this pad:" << node_name << " hasn't been supported";
    return RET_NOT_SUPPORT;
  }

  op->primitive->value.type = schema::PrimitiveType_Pad;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_id, tensors_format, tensors_id_map, tflite_op->inputs[0], tensors_id->size(),
             tflite_tensors.size(), schema::Format::Format_NHWC);
  if (std::strcmp(node_name, "MirrorPad") == 0) {
    AddOpInput(op, tensors_id, tensors_format, tensors_id_map, tflite_op->inputs[1], tensors_id->size(),
               tflite_tensors.size(), schema::Format::Format_NHWC);
  }
  AddOpOutput(op, tensors_id, tensors_format, tensors_id_map, tflite_op->outputs[0], tensors_id->size(),
              tflite_tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tflitePadParser("Pad", new TflitePadParser());
TfliteNodeRegister g_tfliteMirorPadParser("MirrorPad", new TflitePadParser());
}  // namespace lite
}  // namespace mindspore
