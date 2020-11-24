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

#include "tools/converter/parser/tflite/tflite_gather_nd_parser.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {
STATUS TfliteGatherNdParser::Parse(TfliteTensorsInfo *tensors_info, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                   const std::unique_ptr<tflite::ModelT> &tflite_model,
                                   const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "parse TfliteGatherNdParser";
  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  MS_ASSERT(tflite_subgraph != nullptr);
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::GatherNdT> attr = std::make_unique<schema::GatherNdT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  attr->batchDims = 0;

  op->primitive->value.type = schema::PrimitiveType_GatherNd;
  op->primitive->value.value = attr.release();

  for (int input : tflite_op->inputs) {
    AddOpInput(op, tensors_info, input, tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  }
  AddOpOutput(op, tensors_info, tflite_op->outputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}
PrimitiveC *TfliteGatherNdParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return nullptr;
  }

  std::unique_ptr<schema::GatherNdT> attr = std::make_unique<schema::GatherNdT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  attr->batchDims = 0;

  primitive->value.type = schema::PrimitiveType_GatherNd;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteGatherNdParser("GatherND", new TfliteGatherNdParser());
}  // namespace lite
}  // namespace mindspore
