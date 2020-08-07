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
#include "tools/converter/parser/tflite/tflite_sparse_to_dense_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteSparseToDenseParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                                        const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tflite_opset,
                                        schema::CNodeT *op,
                                        TensorCache *tensor_cache, bool quantized_model) {
  MS_LOG(DEBUG) << "parse TfliteSparseToDenseParser";
  std::unique_ptr<schema::SparseToDenseT> attr(new schema::SparseToDenseT());

  if (GetTfliteData(tflite_op->inputs[1], tflite_tensors, tflite_model_buffer, attr->outputShape)) {
    MS_LOG(ERROR) << "sparseToDense -> outputShape get failed";
    return RET_ERROR;
  }
  if (GetTfliteData(tflite_op->inputs[2], tflite_tensors, tflite_model_buffer, attr->sparseValue)) {
    MS_LOG(ERROR) << "sparseToDense -> sparseValue get failed";
    return RET_ERROR;
  }
  if (GetTfliteData(tflite_op->inputs[3], tflite_tensors, tflite_model_buffer, attr->defaultValue)) {
    MS_LOG(ERROR) << "sparseToDense -> defaultValue get failed";
    return RET_ERROR;
  }
  attr->validateIndices = false;
  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_SparseToDense;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

TfliteNodeRegister g_tfliteSparseToDenseParser("SparseToDense", new TfliteSparseToDenseParser());
}  // namespace lite
}  // namespace mindspore

