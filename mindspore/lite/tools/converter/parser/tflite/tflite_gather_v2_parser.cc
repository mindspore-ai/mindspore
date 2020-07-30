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

#include "mindspore/lite/tools/converter/parser/tflite/tflite_gather_v2_parser.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {
STATUS TfliteGatherV2Parser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                              const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                              const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                              const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                              schema::CNodeT *op,
                              TensorCache *tensor_cache,
                              bool quantizedModel) {
  MS_LOG(DEBUG) << "parse TfliteGatherV2Parser";
  std::unique_ptr<schema::GatherT> attr(new schema::GatherT());
  const auto &tflite_attr = tfliteOp->builtin_options.AsGatherOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
  }

  attr->axis = tflite_attr->axis;
  attr->batchDims = 0;    // default

  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Gather;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

TfliteNodeRegister g_tfliteGatherV2Parser("GatherV2", new TfliteGatherV2Parser());
}  // namespace lite
}  // namespace mindspore


