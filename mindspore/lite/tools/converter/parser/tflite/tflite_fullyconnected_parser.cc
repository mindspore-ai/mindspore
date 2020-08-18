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

#include "tools/converter/parser/tflite/tflite_fullyconnected_parser.h"
#include <vector>
#include <memory>
#include <map>
#include <string>

namespace mindspore {
namespace lite {
STATUS TfliteFullyConnectedParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                                         const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                         schema::CNodeT *op,
                                         std::vector<int32_t> *tensors_id,
                                         std::vector<schema::Format> *tensors_format,
                                         std::map<int, int>  *tensors_id_map) {
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
  Split(op->name, &node_name_str, "-");
  const char *node_name = node_name_str.data()->c_str();
  if (std::strcmp(node_name, "FullyConnected") == 0) {
    MS_LOG(DEBUG) << "parse TfliteFullyConnectedParser";
  } else if (std::strcmp(node_name, "FakeQuant") == 0) {
    MS_LOG(DEBUG) << "parse TfliteFakeQuantParser";
  }
  std::unique_ptr<schema::FullConnectionT> attr(new schema::FullConnectionT());

  const auto &tflite_attr = tflite_op->builtin_options.AsFullyConnectedOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name << " attr failed";
    return RET_NULL_PTR;
  }

  attr->hasBias = true;
  attr->axis = 1;
  attr->useAxis = false;
  attr->activationType = GetActivationFunctionType(tflite_attr->fused_activation_function);

  op->primitive->value.type = schema::PrimitiveType_FullConnection;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_id, tensors_format, tensors_id_map,
             tflite_op->inputs[0], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  AddOpInput(op, tensors_id, tensors_format, tensors_id_map,
             tflite_op->inputs[1], tensors_id->size(), tflite_tensors.size(), schema::Format_KHWC);
  AddOpInput(op, tensors_id, tensors_format, tensors_id_map,
             tflite_op->inputs[2], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  AddOpOutput(op, tensors_id, tensors_format, tensors_id_map,
              tflite_op->outputs[0], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteFullyConnectedParser("FullyConnected", new TfliteFullyConnectedParser());
TfliteNodeRegister g_tfliteFakeQuantParser("FakeQuant", new TfliteFakeQuantParser());;
}  // namespace lite
}  // namespace mindspore

