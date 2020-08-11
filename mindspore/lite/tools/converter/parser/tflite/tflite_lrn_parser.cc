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

#include "tools/converter/parser/tflite/tflite_lrn_parser.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {
STATUS TfliteLRNParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
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

  MS_LOG(DEBUG) << "parse TfliteLRNParser";
  std::unique_ptr<schema::LocalResponseNormalizationT> attr(new schema::LocalResponseNormalizationT());

  const auto &tflite_attr = tfliteOp->builtin_options.AsLocalResponseNormalizationOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
    return RET_NULL_PTR;
  }
  attr->depth_radius = tflite_attr->radius;
  attr->alpha = tflite_attr->alpha;
  attr->beta = tflite_attr->beta;
  attr->bias = tflite_attr->bias;

  op->primitive->value.type = schema::PrimitiveType_LocalResponseNormalization;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

TfliteNodeRegister g_tfliteLRNParser("LocalResponseNorm", new TfliteLRNParser());
}  // namespace lite
}  // namespace mindspore


