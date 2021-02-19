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
 * distributed under the License is distributed on an AS
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tools/converter/parser/tflite/tflite_depth_to_space_parser.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {
PrimitiveC *TfliteDepthToSpaceParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  std::unique_ptr<schema::DepthToSpaceT> attr = std::make_unique<schema::DepthToSpaceT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsDepthToSpaceOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op depthtospace attr failed";
    return nullptr;
  }
  attr->blockSize = tflite_attr->block_size;
  attr->format = schema::Format::Format_NHWC;
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_DepthToSpace;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteDepthToSpaceParser(tflite::BuiltinOperator_DEPTH_TO_SPACE, new TfliteDepthToSpaceParser());
}  // namespace lite
}  // namespace mindspore
