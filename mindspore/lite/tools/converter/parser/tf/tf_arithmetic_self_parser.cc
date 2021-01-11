/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/tf/tf_arithmetic_self_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {

template <typename T>
int CreateOperator(const std::unique_ptr<schema::PrimitiveT> &primitive, schema::PrimitiveType type) {
  auto attr = std::make_unique<T>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return RET_NULL_PTR;
  }
  primitive->value.type = type;
  primitive->value.value = attr.release();
  return RET_OK;
}

STATUS TFArithmeticSelfParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     PrimitiveC **primitiveC, std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF ArithmeticParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "New PrimitiveT failed";
    return RET_NULL_PTR;
  }

  int status = RET_ERROR;
  if (tf_op.op() == "Ceil") {
    status = CreateOperator<schema::CeilT>(primitive, schema::PrimitiveType_Ceil);
  } else if (tf_op.op() == "Exp") {
    status = CreateOperator<schema::ExpT>(primitive, schema::PrimitiveType_Exp);
  } else if (tf_op.op() == "Floor") {
    status = CreateOperator<schema::FloorT>(primitive, schema::PrimitiveType_Floor);
  } else if (tf_op.op() == "Log") {
    status = CreateOperator<schema::LogT>(primitive, schema::PrimitiveType_Log);
  } else if (tf_op.op() == "Sqrt") {
    status = CreateOperator<schema::SqrtT>(primitive, schema::PrimitiveType_Sqrt);
  }
  if (status != RET_OK) {
    return status;
  }
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  *output_size = 1;
  status = AddOpInput(tf_op, 0, inputs);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return status;
  }
  return status;
}
TFNodeRegistrar g_tfCeilParser("Ceil", new TFArithmeticSelfParser());
TFNodeRegistrar g_tfExpParser("Exp", new TFArithmeticSelfParser());
TFNodeRegistrar g_tfFloorParser("Floor", new TFArithmeticSelfParser());
TFNodeRegistrar g_tfLogParser("Log", new TFArithmeticSelfParser());
TFNodeRegistrar g_tfSqrtParser("Sqrt", new TFArithmeticSelfParser());
}  // namespace lite
}  // namespace mindspore
