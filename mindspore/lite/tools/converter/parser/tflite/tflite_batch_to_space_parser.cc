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

#include "tools/converter/parser/tflite/tflite_batch_to_space_parser.h"
#include <vector>
#include <memory>
#include <string>

namespace mindspore {
namespace lite {
STATUS TfliteBatchToSpaceParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                                       const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                                       const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                                       schema::CNodeT *op, TensorCache *tensor_cache, bool quantizedModel) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::vector<std::string> node_name_str;
  Split(op->name.data(), &node_name_str, "-");
  const char *node_name = node_name_str.data()->c_str();
  if (std::strcmp(node_name, "BatchToSpace") == 0) {
    MS_LOG(DEBUG) << "parse TfliteBatchToSpaceParser";
  } else if (std::strcmp(node_name, "BatchToSpaceND") == 0) {
    MS_LOG(DEBUG) << "parse TfliteBatchToSpaceNDParser";
    // in tflite
    // blockShape should be a 1D tensor with dimension [spatial_dims_num]
    // crops should be a 2D tensor with dimension [spatial_dims_num, 2]
  }

  std::unique_ptr<schema::BatchToSpaceT> attr(new schema::BatchToSpaceT());

  if (GetTfliteData(tfliteOp->inputs[1], tfliteTensors, tfliteModelBuffer, attr->blockShape)) {
    MS_LOG(ERROR) << "get batchToSpace -> blockShape failed";
    return RET_ERROR;
  }
  if (GetTfliteData(tfliteOp->inputs[2], tfliteTensors, tfliteModelBuffer, attr->crops)) {
    MS_LOG(ERROR) << "get batchToSpace -> crops failed";
    return RET_ERROR;
  }

  op->primitive->value.type = schema::PrimitiveType_BatchToSpace;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

TfliteNodeRegister g_tfliteBatchToSpaceParser("BatchToSpace", new TfliteBatchToSpaceParser());
TfliteNodeRegister g_TfliteBatchToSpaceNDParser("BatchToSpaceND", new TfliteBatchToSpaceNDParser());

}  // namespace lite
}  // namespace mindspore
