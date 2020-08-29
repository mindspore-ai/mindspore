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

#include "tools/converter/parser/tflite/tflite_l2norm_parser.h"
#include <vector>
#include <memory>
#include <map>

namespace mindspore {
namespace lite {
STATUS TfliteL2NormParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                               const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                               const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                               schema::CNodeT *op,
                               std::vector<int32_t> *tensors_id,
                               std::vector<schema::Format> *tensors_format,
                               std::map<int, int>  *tensors_id_map)  {
  MS_LOG(DEBUG) << "parse TfliteL2NormParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::L2NormT> attr = std::make_unique<schema::L2NormT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  if (tflite_op->inputs.empty()) {
    MS_LOG(ERROR) << "the input is null";
    return RET_NULL_PTR;
  }
  auto data_index = tflite_op->inputs[0];
  const auto &data_tensor = tflite_tensors[data_index];
  if (data_tensor == nullptr) {
    MS_LOG(ERROR) << "the input tensor is null";
    return RET_NULL_PTR;
  }

  auto ndim = data_tensor->shape.size();
  std::vector<int32_t> axis;
  axis.reserve(ndim);
  for (size_t i = 0; i < ndim; i++) {
      axis.emplace_back(i);
    }
  attr->axis = axis;
  attr->epsilon = 0.0f;

  op->primitive->value.type = schema::PrimitiveType_L2Norm;
  op->primitive->value.value = attr.release();

  // set input
  AddOpInput(op, tensors_id, tensors_format, tensors_id_map,
               tflite_op->inputs[0], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  AddOpOutput(op, tensors_id, tensors_format, tensors_id_map,
              tflite_op->outputs[0], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteL2NormParser("L2_NORMALIZATION", new TfliteL2NormParser());
}  // namespace lite
}  // namespace mindspore
