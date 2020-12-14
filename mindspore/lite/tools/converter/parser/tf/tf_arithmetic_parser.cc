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
#include "tools/converter/parser/tf/tf_arithmetic_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
STATUS TFArithmeticParser::Parse(const tensorflow::NodeDef &tf_op,
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

  if (tf_op.op() == "Add" || tf_op.op() == "AddV2") {
    auto attr = std::make_unique<schema::AddT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_Add;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "Sub") {
    auto attr = std::make_unique<schema::SubT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_Sub;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "Mul") {
    auto attr = std::make_unique<schema::MulT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_Mul;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "Div" || tf_op.op() == "RealDiv") {
    auto attr = std::make_unique<schema::DivT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_Div;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "Maximum") {
    auto attr = std::make_unique<schema::MaximumT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_Maximum;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "Minimum") {
    auto attr = std::make_unique<schema::MinimumT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_Minimum;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "Greater") {
    auto attr = std::make_unique<schema::GreaterT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_Greater;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "GreaterEqual") {
    auto attr = std::make_unique<schema::GreaterEqualT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_GreaterEqual;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "Less") {
    auto attr = std::make_unique<schema::LessT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_Less;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "LessEqual") {
    auto attr = std::make_unique<schema::LessEqualT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_LessEqual;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "Equal") {
    auto attr = std::make_unique<schema::EqualT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_Equal;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "NotEqual") {
    auto attr = std::make_unique<schema::NotEqualT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_NotEqual;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "FloorMod") {
    auto attr = std::make_unique<schema::FloorModT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_FloorMod;
    primitive->value.value = attr.release();
  } else if (tf_op.op() == "FloorDiv") {
    auto attr = std::make_unique<schema::FloorDivT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr failed";
      return RET_NULL_PTR;
    }
    primitive->value.type = schema::PrimitiveType_FloorDiv;
    primitive->value.value = attr.release();
  }

  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  *output_size = 1;
  auto status = AddOpInput(tf_op, 0, inputs);
  if (status != RET_OK) {
    return status;
  }
  status = AddOpInput(tf_op, 1, inputs);
  return status;
}
TFNodeRegistrar g_tfAddParser("Add", new TFArithmeticParser());
TFNodeRegistrar g_tfAddV2Parser("AddV2", new TFArithmeticParser());
TFNodeRegistrar g_tfSubParser("Sub", new TFArithmeticParser());
TFNodeRegistrar g_tfMulParser("Mul", new TFArithmeticParser());
TFNodeRegistrar g_tfDivParser("Div", new TFArithmeticParser());
TFNodeRegistrar g_tfFloorModParser("FloorMod", new TFArithmeticParser());
TFNodeRegistrar g_tfFloorDivParser("FloorDiv", new TFArithmeticParser());
TFNodeRegistrar g_tfRealDivParser("RealDiv", new TFArithmeticParser());
TFNodeRegistrar g_tfMaximumParser("Maximum", new TFArithmeticParser());
TFNodeRegistrar g_tfMinimumParser("Minimum", new TFArithmeticParser());
TFNodeRegistrar g_tfGreaterParser("Greater", new TFArithmeticParser());
TFNodeRegistrar g_tfGreaterEqualParser("GreaterEqual", new TFArithmeticParser());
TFNodeRegistrar g_tfLessParser("Less", new TFArithmeticParser());
TFNodeRegistrar g_tfLessEqualParser("LessEqual", new TFArithmeticParser());
TFNodeRegistrar g_tfEqualParser("Equal", new TFArithmeticParser());
TFNodeRegistrar g_tfNotEqualParser("NotEqual", new TFArithmeticParser());
}  // namespace lite
}  // namespace mindspore
