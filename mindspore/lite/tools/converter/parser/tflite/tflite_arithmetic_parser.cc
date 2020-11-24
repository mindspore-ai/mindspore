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
#include <string>

namespace mindspore::lite {
PrimitiveC *TfliteDoubleInputOpParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto tflite_op_type = (tflite_model->operator_codes[tflite_op->opcode_index])->builtin_code;
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (tflite_op_type == tflite::BuiltinOperator_ADD) {
    MS_LOG(DEBUG) << "parse TfliteAddParser";
    auto attr = std::make_unique<schema::AddT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    const auto &tfliteAttr = tflite_op->builtin_options.AsAddOptions();
    if (nullptr == tfliteAttr) {
      MS_LOG(ERROR) << "get op: " << tflite_op_type << " attr failed";
      return nullptr;
    }
    attr->activationType = GetActivationFunctionType(tfliteAttr->fused_activation_function);
    primitive->value.type = schema::PrimitiveType_Add;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_SUB) {
    MS_LOG(DEBUG) << "parse TfliteSubParser";
    auto attr = std::make_unique<schema::SubT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    const auto &tfliteAttr = tflite_op->builtin_options.AsSubOptions();
    if (nullptr == tfliteAttr) {
      MS_LOG(ERROR) << "get op: " << tflite_op_type << " attr failed";
      return nullptr;
    }
    attr->activationType = GetActivationFunctionType(tfliteAttr->fused_activation_function);
    primitive->value.type = schema::PrimitiveType_Sub;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_MUL) {
    MS_LOG(DEBUG) << "parse TfliteMulParser";
    auto attr = std::make_unique<schema::MulT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    const auto &tfliteAttr = tflite_op->builtin_options.AsMulOptions();
    if (nullptr == tfliteAttr) {
      MS_LOG(ERROR) << "get op: " << tflite_op_type << " attr failed";
      return nullptr;
    }
    attr->activationType = GetActivationFunctionType(tfliteAttr->fused_activation_function);
    primitive->value.type = schema::PrimitiveType_Mul;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_DIV) {
    MS_LOG(DEBUG) << "parse TfliteDivParser";
    auto attr = std::make_unique<schema::DivT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    const auto &tfliteAttr = tflite_op->builtin_options.AsDivOptions();
    if (nullptr == tfliteAttr) {
      MS_LOG(ERROR) << "get op: " << tflite_op_type << " attr failed";
      return nullptr;
    }
    attr->activationType = GetActivationFunctionType(tfliteAttr->fused_activation_function);
    primitive->value.type = schema::PrimitiveType_Div;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_FLOOR_DIV) {
    MS_LOG(DEBUG) << "parse TfliteFloorDivParser";
    std::unique_ptr<schema::FloorDivT> attr = std::make_unique<schema::FloorDivT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_FloorDiv;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_FLOOR_MOD) {
    MS_LOG(DEBUG) << "parse TfliteFloorModParser";
    auto attr = std::make_unique<schema::FloorModT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_FloorMod;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_SQUARED_DIFFERENCE) {
    MS_LOG(DEBUG) << "parse TfliteSquaredDifferenceParser";
    auto attr = std::make_unique<schema::SquaredDifferenceT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_SquaredDifference;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_POW) {
    MS_LOG(DEBUG) << "parse TflitePowParser";
    auto attr = std::make_unique<schema::PowerT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    attr->power = 1.0f;
    attr->scale = 1.0f;
    attr->shift = 0.0f;
    primitive->value.type = schema::PrimitiveType_Power;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_MAXIMUM) {
    MS_LOG(DEBUG) << "parse TfliteMaximumParser";
    auto attr = std::make_unique<schema::MaximumT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Maximum;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_MINIMUM) {
    MS_LOG(DEBUG) << "parse TfliteMinimumParser";
    auto attr = std::make_unique<schema::MinimumT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Minimum;
    primitive->value.value = attr.release();
  } else {
    MS_LOG(ERROR) << "op hasn't been supported";
    return nullptr;
  }
  return PrimitiveC::Create(primitive.release());
}

PrimitiveC *TfliteSingleInputOpParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto tflite_op_type = (tflite_model->operator_codes[tflite_op->opcode_index])->builtin_code;
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (tflite_op_type == tflite::BuiltinOperator_ABS) {
    MS_LOG(DEBUG) << "parse TfliteAbsParser";
    auto attr = std::make_unique<schema::AbsT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Abs;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_EXP) {
    MS_LOG(DEBUG) << "parse TfliteExpParser";
    auto attr = std::make_unique<schema::ExpT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    attr->base = -1;  // -1 represent base = e
    attr->scale = 1;
    attr->shift = 0;
    primitive->value.type = schema::PrimitiveType_Exp;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_SQRT) {
    MS_LOG(DEBUG) << "parse TfliteSqrtParser";
    auto attr = std::make_unique<schema::SqrtT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Sqrt;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_RSQRT) {
    MS_LOG(DEBUG) << "parse TfliteRsqrtParser";
    auto attr = std::make_unique<schema::RsqrtT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Rsqrt;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_SQUARE) {
    MS_LOG(DEBUG) << "parse TfliteSquareParser";
    auto attr = std::make_unique<schema::SquareT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Square;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_SIN) {
    MS_LOG(DEBUG) << "parse TfliteSinParser";
    auto attr = std::make_unique<schema::SinT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Sin;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_COS) {
    MS_LOG(DEBUG) << "parse TfliteCosParser";
    std::unique_ptr<schema::CosT> attr = std::make_unique<schema::CosT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Cos;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_LOG) {
    MS_LOG(DEBUG) << "parse TfliteLogParser";
    auto attr = std::make_unique<schema::LogT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Log;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_ROUND) {
    MS_LOG(DEBUG) << "parse TfliteRoundParser";
    auto attr = std::make_unique<schema::RoundT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Round;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_CEIL) {
    MS_LOG(DEBUG) << "parse TfliteCeilParser";
    auto attr = std::make_unique<schema::CeilT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Ceil;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_FLOOR) {
    MS_LOG(DEBUG) << "parse TfliteFloorParser";
    auto attr = std::make_unique<schema::FloorT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Floor;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_NEG) {
    MS_LOG(DEBUG) << "parse TfliteNegParser";
    auto attr = std::make_unique<schema::NegT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Neg;
    primitive->value.value = attr.release();
  }
  return PrimitiveC::Create(primitive.release());
}

PrimitiveC *TfliteCompareOpParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                      const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto tflite_op_type = (tflite_model->operator_codes[tflite_op->opcode_index])->builtin_code;
  auto primitive = std::make_unique<schema::PrimitiveT>();

  if (tflite_op_type == tflite::BuiltinOperator_EQUAL) {
    MS_LOG(DEBUG) << "parse TfliteEqualParser";
    std::unique_ptr<schema::EqualT> attr = std::make_unique<schema::EqualT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Equal;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_NOT_EQUAL) {
    MS_LOG(DEBUG) << "parse TfliteNotEqualParser";
    std::unique_ptr<schema::NotEqualT> attr = std::make_unique<schema::NotEqualT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_NotEqual;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_GREATER) {
    MS_LOG(DEBUG) << "parse TfliteGreaterParser";
    std::unique_ptr<schema::GreaterT> attr = std::make_unique<schema::GreaterT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Greater;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_GREATER_EQUAL) {
    MS_LOG(DEBUG) << "parse TfliteGreaterEqualParser";
    std::unique_ptr<schema::GreaterEqualT> attr = std::make_unique<schema::GreaterEqualT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_GreaterEqual;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_LESS) {
    MS_LOG(DEBUG) << "parse TfliteLessParser";
    std::unique_ptr<schema::LessT> attr = std::make_unique<schema::LessT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_Less;
    primitive->value.value = attr.release();
  } else if (tflite_op_type == tflite::BuiltinOperator_LESS_EQUAL) {
    MS_LOG(DEBUG) << "parse TfliteLessEqualParser";
    std::unique_ptr<schema::LessEqualT> attr = std::make_unique<schema::LessEqualT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    primitive->value.type = schema::PrimitiveType_LessEqual;
    primitive->value.value = attr.release();
  }
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteAddParser(tflite::BuiltinOperator_ADD, new TfliteDoubleInputOpParser());
TfliteNodeRegister g_tfliteSubParser(tflite::BuiltinOperator_SUB, new TfliteDoubleInputOpParser());
TfliteNodeRegister g_TfliteMulParser(tflite::BuiltinOperator_MUL, new TfliteDoubleInputOpParser());
TfliteNodeRegister g_TfliteDivParser(tflite::BuiltinOperator_DIV, new TfliteDoubleInputOpParser());
TfliteNodeRegister g_tfliteFloorDivParser(tflite::BuiltinOperator_FLOOR_DIV, new TfliteDoubleInputOpParser());
TfliteNodeRegister g_tfliteFloorModParser(tflite::BuiltinOperator_FLOOR_MOD, new TfliteDoubleInputOpParser());
TfliteNodeRegister g_TflitePowParser(tflite::BuiltinOperator_POW, new TfliteDoubleInputOpParser());
TfliteNodeRegister g_tfliteSquaredDifferenceParser(tflite::BuiltinOperator_SQUARED_DIFFERENCE,
                                                   new TfliteDoubleInputOpParser());
TfliteNodeRegister g_TfliteMaximumParser(tflite::BuiltinOperator_MAXIMUM, new TfliteDoubleInputOpParser());
TfliteNodeRegister g_TfliteMinimumParser(tflite::BuiltinOperator_MINIMUM, new TfliteDoubleInputOpParser());

TfliteNodeRegister g_TfliteAbsParser(tflite::BuiltinOperator_ABS, new TfliteSingleInputOpParser());
TfliteNodeRegister g_TfliteExpParser(tflite::BuiltinOperator_EXP, new TfliteSingleInputOpParser());
TfliteNodeRegister g_TfliteSqrtParser(tflite::BuiltinOperator_SQRT, new TfliteSingleInputOpParser());
TfliteNodeRegister g_tfliteRsqrtParser(tflite::BuiltinOperator_RSQRT, new TfliteSingleInputOpParser());
TfliteNodeRegister g_TfliteSquareParser(tflite::BuiltinOperator_SQUARE, new TfliteSingleInputOpParser());
TfliteNodeRegister g_TfliteSinParser(tflite::BuiltinOperator_SIN, new TfliteSingleInputOpParser());
TfliteNodeRegister g_TfliteCosParser(tflite::BuiltinOperator_COS, new TfliteSingleInputOpParser());
TfliteNodeRegister g_TfliteLogParser(tflite::BuiltinOperator_LOG, new TfliteSingleInputOpParser());
TfliteNodeRegister g_tfliteRoundParser(tflite::BuiltinOperator_ROUND, new TfliteSingleInputOpParser());
TfliteNodeRegister g_TfliteCeilParser(tflite::BuiltinOperator_CEIL, new TfliteSingleInputOpParser());
TfliteNodeRegister g_tfliteFloorParser(tflite::BuiltinOperator_FLOOR, new TfliteSingleInputOpParser());
TfliteNodeRegister g_tfliteNegParser(tflite::BuiltinOperator_NEG, new TfliteSingleInputOpParser());

TfliteNodeRegister g_tfliteEqualParser(tflite::BuiltinOperator_EQUAL, new TfliteCompareOpParser());
TfliteNodeRegister g_tfliteNotEqualParser(tflite::BuiltinOperator_NOT_EQUAL, new TfliteCompareOpParser());
TfliteNodeRegister g_tfliteGreaterEParser(tflite::BuiltinOperator_GREATER, new TfliteCompareOpParser());
TfliteNodeRegister g_tfliteGreaterEqualParser(tflite::BuiltinOperator_GREATER_EQUAL, new TfliteCompareOpParser());
TfliteNodeRegister g_tfliteLessParser(tflite::BuiltinOperator_LESS, new TfliteCompareOpParser());
TfliteNodeRegister g_tfliteLessEqualParser(tflite::BuiltinOperator_LESS_EQUAL, new TfliteCompareOpParser());
}  // namespace mindspore::lite
