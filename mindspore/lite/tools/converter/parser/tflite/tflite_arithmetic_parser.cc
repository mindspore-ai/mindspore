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

#include "tools/converter/parser/tflite/tflite_arithmetic_parser.h"
#include <vector>
#include <memory>
#include "ops/abs.h"
#include "ops/cos.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/fusion/sub_fusion.h"
#include "ops/fusion/exp_fusion.h"
#include "ops/fusion/pow_fusion.h"
#include "ops/squared_difference.h"
#include "ops/square.h"
#include "ops/sqrt.h"
#include "ops/rsqrt.h"
#include "ops/sin.h"
#include "ops/log.h"
#include "ops/round.h"
#include "ops/neg.h"
#include "ops/maximum.h"
#include "ops/minimum.h"
#include "ops/floor.h"
#include "ops/floor_div.h"
#include "ops/floor_mod.h"
#include "ops/ceil.h"
#include "ops/equal.h"
#include "ops/greater.h"
#include "ops/greater_equal.h"
#include "ops/less.h"
#include "ops/less_equal.h"
#include "ops/not_equal.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteAddParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::AddFusion();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new AddFusion failed";
    return nullptr;
  }

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsAddOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get AddFusion attr failed";
    return nullptr;
  }
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  return prim;
}

ops::PrimitiveC *TfliteMulParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::MulFusion();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new MulFusion failed";
    return nullptr;
  }

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsMulOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get MulFusion attr failed";
    return nullptr;
  }
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  return prim;
}

ops::PrimitiveC *TfliteDivParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::DivFusion();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new DivFusion failed";
    return nullptr;
  }

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsDivOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get DivFusion attr failed";
    return nullptr;
  }
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  return prim;
}

ops::PrimitiveC *TfliteSubParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::SubFusion();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new SubFusion failed";
    return nullptr;
  }

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsSubOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get SubFusion attr failed";
    return nullptr;
  }
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  return prim;
}

ops::PrimitiveC *TfliteFloorDivParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                             const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::FloorDiv();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new FloorDiv failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteFloorModParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                             const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::FloorMod();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new FloorMod failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TflitePowParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::PowFusion();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new PowFusion failed";
    return nullptr;
  }

  prim->set_scale(1.0);
  prim->set_shift(0.0);

  return prim;
}

ops::PrimitiveC *TfliteSquaredDifferenceParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                      const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::SquaredDifference();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new SquaredDifference failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteMaximumParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                            const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Maximum();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Maximum failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteMinimumParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                            const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Minimum();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Minimum failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteAbsParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Abs();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Abs failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteCosParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Cos();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Cos failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteFloorParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Floor();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Floor failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteExpParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::ExpFusion();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new ExpFusion failed";
    return nullptr;
  }

  prim->set_base(-1.0);
  prim->set_scale(1.0);
  prim->set_shift(0.0);

  return prim;
}

ops::PrimitiveC *TfliteCeilParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Ceil();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Ceil failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteLogParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Log();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Log failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteRoundParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Round();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Round failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteSqrtParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Sqrt();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Sqrt failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteRsqrtParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Rsqrt();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Rsqrt failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteSquareParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                           const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Square();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Square failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteSinParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Sin();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Sin failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteNegParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Neg();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Neg failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteEqualParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Equal();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Equal failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteNotEqualParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                             const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::NotEqual();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new NotEqual failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteGreaterParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                            const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Greater();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Greater failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteGreaterEqualParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                 const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::GreaterEqual();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new GreaterEqual failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteLessParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::Less();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Less failed";
    return nullptr;
  }

  return prim;
}

ops::PrimitiveC *TfliteLessEqualParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                              const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = new (std::nothrow) ops::LessEqual();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new LessEqual failed";
    return nullptr;
  }

  return prim;
}

TfliteNodeRegister g_tfliteAddParser(tflite::BuiltinOperator_ADD, new TfliteAddParser());
TfliteNodeRegister g_tfliteSubParser(tflite::BuiltinOperator_SUB, new TfliteSubParser());
TfliteNodeRegister g_TfliteMulParser(tflite::BuiltinOperator_MUL, new TfliteMulParser());
TfliteNodeRegister g_TfliteDivParser(tflite::BuiltinOperator_DIV, new TfliteDivParser());
TfliteNodeRegister g_tfliteFloorDivParser(tflite::BuiltinOperator_FLOOR_DIV, new TfliteFloorDivParser());
TfliteNodeRegister g_tfliteFloorModParser(tflite::BuiltinOperator_FLOOR_MOD, new TfliteFloorModParser());
TfliteNodeRegister g_TflitePowParser(tflite::BuiltinOperator_POW, new TflitePowParser());
TfliteNodeRegister g_tfliteSquaredDifferenceParser(tflite::BuiltinOperator_SQUARED_DIFFERENCE,
                                                   new TfliteSquaredDifferenceParser());
TfliteNodeRegister g_TfliteMaximumParser(tflite::BuiltinOperator_MAXIMUM, new TfliteMaximumParser());
TfliteNodeRegister g_TfliteMinimumParser(tflite::BuiltinOperator_MINIMUM, new TfliteMinimumParser());
TfliteNodeRegister g_TfliteAbsParser(tflite::BuiltinOperator_ABS, new TfliteAbsParser());
TfliteNodeRegister g_TfliteExpParser(tflite::BuiltinOperator_EXP, new TfliteExpParser());
TfliteNodeRegister g_TfliteSqrtParser(tflite::BuiltinOperator_SQRT, new TfliteSqrtParser());
TfliteNodeRegister g_tfliteRsqrtParser(tflite::BuiltinOperator_RSQRT, new TfliteRsqrtParser());
TfliteNodeRegister g_TfliteSquareParser(tflite::BuiltinOperator_SQUARE, new TfliteSquareParser());
TfliteNodeRegister g_TfliteSinParser(tflite::BuiltinOperator_SIN, new TfliteSinParser());
TfliteNodeRegister g_TfliteCosParser(tflite::BuiltinOperator_COS, new TfliteCosParser());
TfliteNodeRegister g_TfliteLogParser(tflite::BuiltinOperator_LOG, new TfliteLogParser());
TfliteNodeRegister g_tfliteRoundParser(tflite::BuiltinOperator_ROUND, new TfliteRoundParser());
TfliteNodeRegister g_TfliteCeilParser(tflite::BuiltinOperator_CEIL, new TfliteCeilParser());
TfliteNodeRegister g_tfliteFloorParser(tflite::BuiltinOperator_FLOOR, new TfliteFloorParser());
TfliteNodeRegister g_tfliteNegParser(tflite::BuiltinOperator_NEG, new TfliteNegParser());
TfliteNodeRegister g_tfliteEqualParser(tflite::BuiltinOperator_EQUAL, new TfliteEqualParser());
TfliteNodeRegister g_tfliteNotEqualParser(tflite::BuiltinOperator_NOT_EQUAL, new TfliteNotEqualParser());
TfliteNodeRegister g_tfliteGreaterEParser(tflite::BuiltinOperator_GREATER, new TfliteGreaterParser());
TfliteNodeRegister g_tfliteGreaterEqualParser(tflite::BuiltinOperator_GREATER_EQUAL, new TfliteGreaterEqualParser());
TfliteNodeRegister g_tfliteLessParser(tflite::BuiltinOperator_LESS, new TfliteLessParser());
TfliteNodeRegister g_tfliteLessEqualParser(tflite::BuiltinOperator_LESS_EQUAL, new TfliteLessEqualParser());
}  // namespace lite
}  // namespace mindspore
