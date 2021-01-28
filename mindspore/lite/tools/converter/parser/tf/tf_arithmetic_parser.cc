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
    auto prim = std::make_unique<ops::AddFusion>();
    return prim.release();
  } else if (tf_op.op() == "Sub") {
    auto prim = std::make_unique<ops::SubFusion>();
    return prim.release();
  } else if (tf_op.op() == "Mul") {
    auto prim = std::make_unique<ops::MulFusion>();
    return prim.release();
  } else if (tf_op.op() == "Div" || tf_op.op() == "RealDiv") {
    auto prim = std::make_unique<ops::DivFusion>();
    return prim.release();
  } else if (tf_op.op() == "Maximum") {
    auto prim = std::make_unique<ops::Maximum>();
    return prim.release();
  } else if (tf_op.op() == "Minimum") {
    auto prim = std::make_unique<ops::Minimum>();
    return prim.release();
  } else if (tf_op.op() == "Greater") {
    auto prim = std::make_unique<ops::Greater>();
    return prim.release();
  } else if (tf_op.op() == "GreaterEqual") {
    auto prim = std::make_unique<ops::GreaterEqual>();
    return prim.release();
  } else if (tf_op.op() == "Less") {
    auto prim = std::make_unique<ops::Less>();
    return prim.release();
  } else if (tf_op.op() == "LessEqual") {
    auto prim = std::make_unique<ops::LessEqual>();
    return prim.release();
  } else if (tf_op.op() == "Equal") {
    auto prim = std::make_unique<ops::Equal>();
    return prim.release();
  } else if (tf_op.op() == "NotEqual") {
    auto prim = std::make_unique<ops::NotEqual>();
    return prim.release();
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
