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

#ifndef PREDICT_TFLITE_DEPTHWISE_CONV_PARSER_H
#define PREDICT_TFLITE_DEPTHWISE_CONV_PARSER_H

#include <vector>
#include <memory>
#include "tools/converter/parser/tflite/tflite_node_parser.h"
#include "tools/converter/parser/tflite/tflite_node_parser_registry.h"

namespace mindspore {
namespace lite {
class TfliteDepthwiseConv2DParser : public TfliteNodeParser {
 public:
  TfliteDepthwiseConv2DParser() : TfliteNodeParser("DepthwiseConv2D") {}

  STATUS Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
               const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
               const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
               const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
               schema::CNodeT *op, TensorCache *tensor_cache, bool quantizedModel) override;

 private:
  STATUS ParseGroupDepthwiseConv(schema::CNodeT *op,
                                 const std::unique_ptr<schema::DepthwiseConv2DT> &attr,
                                 const std::unique_ptr<tflite::TensorT> &weightTensor,
                                 TensorCache *tensor_cache);
};
}  // namespace lite
}  // namespace mindspore

#endif  // PREDICT_TFLITE_CONV_PARSER_H

