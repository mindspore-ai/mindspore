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

#include "tools/converter/parser/tflite/tflite_logical_parser.h"
#include <vector>
#include <memory>
#include <string>

namespace mindspore {
namespace lite {
STATUS TfliteLogicalParser::Parse(TfliteTensorsInfo *tensors_info, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                  const std::unique_ptr<tflite::ModelT> &tflite_model,
                                  const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
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

  std::vector<std::string> node_name_str;
  Split(op->name, &node_name_str, "-");
  const char *node_name = node_name_str.data()->c_str();
  if (std::strcmp(node_name, "LogicalAnd") == 0) {
    MS_LOG(DEBUG) << "parse TfliteLogicalAndParser";
    std::unique_ptr<schema::LogicalAndT> attr = std::make_unique<schema::LogicalAndT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return RET_NULL_PTR;
    }
    op->primitive->value.type = schema::PrimitiveType_LogicalAnd;
    op->primitive->value.value = attr.release();
  } else if (std::strcmp(node_name, "LogicalNot") == 0) {
    MS_LOG(DEBUG) << "parse TfliteLogicalNotParser";
    std::unique_ptr<schema::LogicalNotT> attr = std::make_unique<schema::LogicalNotT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return RET_NULL_PTR;
    }
    op->primitive->value.type = schema::PrimitiveType_LogicalNot;
    op->primitive->value.value = attr.release();
  } else if (std::strcmp(node_name, "LogicalOr") == 0) {
    MS_LOG(DEBUG) << "parse TfliteLogicalOrParser";
    std::unique_ptr<schema::LogicalOrT> attr = std::make_unique<schema::LogicalOrT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return RET_NULL_PTR;
    }
    op->primitive->value.type = schema::PrimitiveType_LogicalOr;
    op->primitive->value.value = attr.release();
  } else {
    MS_LOG(ERROR) << node_name << " hasn't been supported";
    return RET_NOT_FIND_OP;
  }

  for (int input : tflite_op->inputs) {
    AddOpInput(op, tensors_info, input, tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  }
  AddOpOutput(op, tensors_info, tflite_op->outputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}
PrimitiveC *TfliteLogicalParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                    const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is null";
    return nullptr;
  }
  auto tflite_op_type = (tflite_model->operator_codes[tflite_op->opcode_index])->builtin_code;
  if (tflite_op_type == tflite::BuiltinOperator_LOGICAL_AND) {
    MS_LOG(DEBUG) << "parse TfliteLogicalAndParser";
    std::unique_ptr<schema::LogicalAndT> attr = std::make_unique<schema::LogicalAndT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_LogicalAnd;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_LOGICAL_NOT) {
    MS_LOG(DEBUG) << "parse TfliteLogicalNotParser";
    std::unique_ptr<schema::LogicalNotT> attr = std::make_unique<schema::LogicalNotT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_LogicalNot;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_LOGICAL_OR) {
    MS_LOG(DEBUG) << "parse TfliteLogicalOrParser";
    std::unique_ptr<schema::LogicalOrT> attr = std::make_unique<schema::LogicalOrT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_LogicalOr;
    primitive->value.value = attr.release();
  }
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteLogicalAndParser("LogicalAnd", new TfliteLogicalParser());
TfliteNodeRegister g_tfliteLogicalNotParser("LogicalNot", new TfliteLogicalParser());
TfliteNodeRegister g_tfliteLogicalOrParser("LogicalOr", new TfliteLogicalParser());
}  // namespace lite
}  // namespace mindspore
