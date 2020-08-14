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

#include "tools/converter/parser/tflite/tflite_gather_parser.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {
STATUS TfliteGatherParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                              const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                              const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                              const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                              schema::CNodeT *op,
                              TensorCache *tensor_cache,
                              bool quantizedModel) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  MS_LOG(DEBUG) << "parse TfliteGatherParser";
  std::unique_ptr<schema::GatherT> attr(new schema::GatherT());

  const auto &tflite_attr = tfliteOp->builtin_options.AsGatherOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
    return RET_NULL_PTR;
  }
  attr->axis = tflite_attr->axis;

  attr->batchDims = 0;

  auto y_index = tfliteOp->inputs[1];
  const auto &y_tensor = tfliteTensors[y_index];
  if (y_tensor == nullptr) {
    MS_LOG(ERROR) << "the second input is null";
    return RET_NULL_PTR;
  }
  auto &y_data = tfliteModelBuffer.at(y_tensor->buffer);
  if (y_data == nullptr) {
    MS_LOG(ERROR) << "the data of the second input is null";
    return RET_NULL_PTR;
  }
  if (!y_data->data.empty()) {
    std::vector<tflite::TensorT *> y_tensors{y_tensor.get()};
    if (RET_OK != ParseTensor(y_tensors, tfliteModelBuffer, tensor_cache, TF_CONST, false)) {
      MS_LOG(ERROR) << "parse the second tensor failed";
      return RET_ERROR;
    }
  }

  op->primitive->value.type = schema::PrimitiveType_Gather;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

TfliteNodeRegister g_tfliteGatherParser("Gather", new TfliteGatherParser());
}  // namespace lite
}  // namespace mindspore


