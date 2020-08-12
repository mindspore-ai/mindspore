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

#include "tools/converter/parser/onnx/onnx_node_parser.h"

namespace mindspore {
namespace lite {
schema::PadMode OnnxNodeParser::GetOnnxPadMode(const onnx::AttributeProto &onnx_node_attr) {
  if (onnx_node_attr.s() == "NOTSET") {
    return schema::PadMode_NOTSET;
  } else if (onnx_node_attr.s() == "SAME_UPPER" || onnx_node_attr.s() == "SAME_LOWER") {
    return schema::PadMode_SAME;
  } else if (onnx_node_attr.s() == "VALID") {
    return schema::PadMode_VALID;
  } else {
    // MS_LOGE("unsupported padMode");
    return schema::PadMode_NOTSET;
  }
}
}  // namespace lite
}  // namespace mindspore

