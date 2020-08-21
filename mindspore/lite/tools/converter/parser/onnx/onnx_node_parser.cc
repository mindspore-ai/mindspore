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
#include <vector>

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
    MS_LOG(ERROR) << "unsupported padMode";
    return schema::PadMode_NOTSET;
  }
}

void OnnxNodeParser::Split(const std::string &src_str,
                           std::vector<std::string> *dst_str,
                           const std::string &chr) {
  std::string ::size_type p1 = 0, p2 = src_str.find(chr);
  while (std::string::npos != p2) {
    dst_str->push_back(src_str.substr(p1, p2 - p1));
    p1 = p2 + chr.size();
    p2 = src_str.find(chr, p1);
  }
  if (p1 != src_str.length()) {
    dst_str->push_back(src_str.substr(p1));
  }
}
}  // namespace lite
}  // namespace mindspore

