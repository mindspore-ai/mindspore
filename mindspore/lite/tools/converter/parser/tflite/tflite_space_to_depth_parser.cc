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

#include <vector>
#include <memory>
#include "tools/converter/parser/tflite/tflite_space_to_depth_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteSpaceToDepthParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                       const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                                       const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tflite_opset,
                                       schema::CNodeT *op,
                                       TensorCache *tensor_cache, bool quantized_model) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  MS_LOG(DEBUG) << "parse TfliteSpaceToDepthParser";
  std::unique_ptr<schema::SpaceToDepthT> attr(new schema::SpaceToDepthT());

  const auto &tflite_attr = tflite_op->builtin_options.AsSpaceToDepthOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op:" << op->name.c_str() << " attr failed";
    return RET_NULL_PTR;
  }
  attr->blockSize = tflite_attr->block_size;

  attr->format = schema::Format_NHWC;

  op->primitive->value.type = schema::PrimitiveType_SpaceToDepth;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

TfliteNodeRegister g_tfliteSpaceToDepthParser("SpaceToDepth", new TfliteSpaceToDepthParser());
}  // namespace lite
}  // namespace mindspore
