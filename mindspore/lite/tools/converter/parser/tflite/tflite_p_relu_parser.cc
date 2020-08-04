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

#include <memory>
#include <vector>
#include "tools/converter/parser/tflite/tflite_p_relu_parser.h"

namespace mindspore {
namespace lite {
STATUS TflitePreluParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                                const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tflite_opset,
                                schema::CNodeT *op, TensorCache *tensor_cache, bool quantized_model) {
  MS_LOG(DEBUG) << "paser TflitePreluParser";
  std::unique_ptr<schema::PreluT> attr(new schema::PreluT());

  if (GetTfliteData(tflite_op->inputs[1], tflite_tensors, tflite_model_buffer, attr->slope)) {
    MS_LOG(ERROR) << "get pRelu -> slope failed";
    return RET_ERROR;
  }

  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Prelu;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

TfliteNodeRegister g_tflitePreluParser("Prelu", new TflitePreluParser());
}  // namespace lite
}  // namespace mindspore
