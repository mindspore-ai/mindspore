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
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/greater.h"
#include "ops/greater_equal.h"
#include "ops/less.h"
#include "ops/less_equal.h"
#include "ops/equal.h"
#include "ops/maximum.h"
#include "ops/minimum.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/not_equal.h"
#include "ops/fusion/sub_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFArithmeticParser::Parse(const tensorflow::NodeDef &tf_op,
                                           const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                           std::vector<std::string> *inputs, int *output_size) {
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  if (tf_op.op() == "Add" || tf_op.op() == "AddV2") {
    auto primitive_c = new (std::nothrow) ops::AddFusion;
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "new AddFusion failed";
      return nullptr;
    }
    return primitive_c;
  } else if (tf_op.op() == "Sub") {
    auto primitive_c = new (std::nothrow) ops::SubFusion;
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "new SubFusion failed";
      return nullptr;
    }
    return primitive_c;
  } else if (tf_op.op() == "Mul") {
    auto primitive_c = new (std::nothrow) ops::MulFusion;
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "new MulFusion failed";
      return nullptr;
    }
    return primitive_c;
  } else if (tf_op.op() == "Div" || tf_op.op() == "RealDiv") {
    auto primitive_c = new (std::nothrow) ops::DivFusion;
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "new DivFusion failed";
      return nullptr;
    }
    return primitive_c;
  } else if (tf_op.op() == "Maximum") {
    auto primitive_c = new (std::nothrow) ops::Maximum;
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "new Maximum failed";
      return nullptr;
    }
    return primitive_c;
  } else if (tf_op.op() == "Minimum") {
    auto primitive_c = new (std::nothrow) ops::Minimum;
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "new Minimum failed";
      return nullptr;
    }
    return primitive_c;
  } else if (tf_op.op() == "Greater") {
    auto primitive_c = new (std::nothrow) ops::Greater;
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "new Greater failed";
      return nullptr;
    }
    return primitive_c;
  } else if (tf_op.op() == "GreaterEqual") {
    auto primitive_c = new (std::nothrow) ops::GreaterEqual;
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "new GreaterEqual failed";
      return nullptr;
    }
    return primitive_c;
  } else if (tf_op.op() == "Less") {
    auto primitive_c = new (std::nothrow) ops::Less;
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "new Less failed";
      return nullptr;
    }
    return primitive_c;
  } else if (tf_op.op() == "LessEqual") {
    auto primitive_c = new (std::nothrow) ops::LessEqual;
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "new LessEqual failed";
      return nullptr;
    }
    return primitive_c;
  } else if (tf_op.op() == "Equal") {
    auto primitive_c = new (std::nothrow) ops::Equal;
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "new Equal failed";
      return nullptr;
    }
    return primitive_c;
  } else if (tf_op.op() == "NotEqual") {
    auto primitive_c = new (std::nothrow) ops::NotEqual;
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "new NotEqual failed";
      return nullptr;
    }
    return primitive_c;
  }
  return nullptr;
}

TFNodeRegistrar g_tfAddParser("Add", new TFArithmeticParser());
TFNodeRegistrar g_tfAddV2Parser("AddV2", new TFArithmeticParser());
TFNodeRegistrar g_tfSubParser("Sub", new TFArithmeticParser());
TFNodeRegistrar g_tfMulParser("Mul", new TFArithmeticParser());
TFNodeRegistrar g_tfDivParser("Div", new TFArithmeticParser());
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
