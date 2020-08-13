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

namespace mindspore {
namespace lite {
STATUS TfliteDoubleInputOpParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
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

  if (std::strcmp(node_name, "Add") == 0
      || std::strcmp(node_name, "Sub") == 0
      || std::strcmp(node_name, "Mul") == 0
      || std::strcmp(node_name, "Div") == 0) {
    auto x_index = tfliteOp->inputs[0];
    const auto &x_tensor = tfliteTensors[x_index];
    if (x_tensor == nullptr) {
      MS_LOG(ERROR) << "the first input is null";
      return RET_NULL_PTR;
    }
    auto &x_data = tfliteModelBuffer.at(x_tensor->buffer);
    if (x_data == nullptr) {
      MS_LOG(ERROR) << "the data of the first input is null";
      return RET_NULL_PTR;
    }
    if (!x_data->data.empty()) {
      std::vector<tflite::TensorT *> x_tensors{x_tensor.get()};
      if (RET_OK != ParseTensor(x_tensors, tfliteModelBuffer, tensor_cache, TF_CONST, false)) {
        MS_LOG(ERROR) << "parse the first tensor failed";
        return RET_ERROR;
      }
    }

    auto y_index = tfliteOp->inputs[1];
    const auto &y_tensor = tfliteTensors[y_index];
    if (y_tensor == nullptr) {
      MS_LOG(ERROR) << "the second input is null";
      return RET_NULL_PTR;
    }
    auto &y_data = tfliteModelBuffer.at(y_tensor->buffer);
    if (y_data == nullptr) {
      MS_LOG(ERROR) << "the data of the second input is null";
      return RET_NULL_PTR;
    }
    if (!y_data->data.empty()) {
      std::vector<tflite::TensorT *> y_tensors{y_tensor.get()};
      if (RET_OK != ParseTensor(y_tensors, tfliteModelBuffer, tensor_cache, TF_CONST, false)) {
        MS_LOG(ERROR) << "parse the second tensor failed";
        return RET_ERROR;
      }
    }

    if (std::strcmp(node_name, "Add") == 0) {
      MS_LOG(DEBUG) << "parse TfliteAddParser";
      std::unique_ptr<schema::AddT> attr(new schema::AddT());
      const auto &tfliteAttr = tfliteOp->builtin_options.AsAddOptions();
      if (nullptr == tfliteAttr) {
        MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
        return RET_NULL_PTR;
      }
      attr->activationType = GetActivationFunctionType(tfliteAttr->fused_activation_function);
      op->primitive->value.type = schema::PrimitiveType_Add;
      op->primitive->value.value = attr.release();
      return RET_OK;
    } else if (std::strcmp(node_name, "Sub") == 0) {
      MS_LOG(DEBUG) << "parse TfliteSubParser";
      std::unique_ptr<schema::SubT> attr(new schema::SubT());
      const auto &tfliteAttr = tfliteOp->builtin_options.AsSubOptions();
      if (nullptr == tfliteAttr) {
        MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
        return RET_NULL_PTR;
      }
      attr->activationType = GetActivationFunctionType(tfliteAttr->fused_activation_function);
      op->primitive->value.type = schema::PrimitiveType_Sub;
      op->primitive->value.value = attr.release();
      return RET_OK;
    } else if (std::strcmp(node_name, "Mul") == 0) {
      MS_LOG(DEBUG) << "parse TfliteMulParser";
      std::unique_ptr<schema::MulT> attr(new schema::MulT());
      const auto &tfliteAttr = tfliteOp->builtin_options.AsMulOptions();
      if (nullptr == tfliteAttr) {
        MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
        return RET_NULL_PTR;
      }
      attr->activationType = GetActivationFunctionType(tfliteAttr->fused_activation_function);
      op->primitive->value.type = schema::PrimitiveType_Mul;
      op->primitive->value.value = attr.release();
      return RET_OK;
    } else if (std::strcmp(node_name, "Div") == 0) {
      MS_LOG(DEBUG) << "parse TfliteDivParser";
      std::unique_ptr<schema::DivT> attr(new schema::DivT());
      const auto &tfliteAttr = tfliteOp->builtin_options.AsDivOptions();
      if (nullptr == tfliteAttr) {
        MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
        return RET_NULL_PTR;
      }
      attr->activationType = GetActivationFunctionType(tfliteAttr->fused_activation_function);
      op->primitive->value.type = schema::PrimitiveType_Div;
      op->primitive->value.value = attr.release();
      return RET_OK;
    }
  } else if (std::strcmp(node_name, "FloorDiv") == 0) {
    MS_LOG(DEBUG) << "parse TfliteFloorDivParser";
    std::unique_ptr<schema::FloorDivT> attr(new schema::FloorDivT());
    op->primitive->value.type = schema::PrimitiveType_FloorDiv;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "FloorMod") == 0) {
    MS_LOG(DEBUG) << "parse TfliteFloorModParser";
    std::unique_ptr<schema::FloorModT> attr(new schema::FloorModT());
    op->primitive->value.type = schema::PrimitiveType_FloorMod;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "RealDiv") == 0) {
    MS_LOG(DEBUG) << "parse TfliteRealDivParser";
    std::unique_ptr<schema::RealDivT> attr(new schema::RealDivT());
    op->primitive->value.type = schema::PrimitiveType_RealDiv;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "SquaredDifference") == 0) {
    MS_LOG(DEBUG) << "parse TfliteSquaredDifferenceParser";
    std::unique_ptr<schema::SquaredDifferenceT> attr(new schema::SquaredDifferenceT());
    op->primitive->value.type = schema::PrimitiveType_SquaredDifference;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Pow") == 0) {
    MS_LOG(DEBUG) << "parse TflitePowParser";
    std::unique_ptr<schema::PowerT> attr(new schema::PowerT());
    attr->power = 0.0f;
    attr->scale = 1.0f;
    attr->shift = 0.0f;
    op->primitive->value.type = schema::PrimitiveType_Power;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Maximum") == 0) {
    MS_LOG(DEBUG) << "parse TfliteMaximumParser";
    std::unique_ptr<schema::MaximumT> attr(new schema::MaximumT());
    op->primitive->value.type = schema::PrimitiveType_Maximum;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Minimum") == 0) {
    MS_LOG(DEBUG) << "parse TfliteMinimumParser";
    std::unique_ptr<schema::MinimumT> attr(new schema::MinimumT());
    op->primitive->value.type = schema::PrimitiveType_Minimum;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "wrong op type";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TfliteSingleInputOpParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
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
  if (std::strcmp(node_name, "Abs") == 0) {
    MS_LOG(DEBUG) << "parse TfliteAbsParser";
    std::unique_ptr<schema::AbsT> attr(new schema::AbsT());
    op->primitive->value.type = schema::PrimitiveType_Abs;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Exp") == 0) {
    MS_LOG(DEBUG) << "parse TfliteExpParser";
    std::unique_ptr<schema::ExpT> attr(new schema::ExpT());
    op->primitive->value.type = schema::PrimitiveType_Exp;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Sqrt") == 0) {
    MS_LOG(DEBUG) << "parse TfliteSqrtParser";
    std::unique_ptr<schema::SqrtT> attr(new schema::SqrtT());
    op->primitive->value.type = schema::PrimitiveType_Sqrt;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Rsqrt") == 0) {
    MS_LOG(DEBUG) << "parse TfliteRsqrtParser";
    std::unique_ptr<schema::RsqrtT> attr(new schema::RsqrtT());
    op->primitive->value.type = schema::PrimitiveType_Rsqrt;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Square") == 0) {
    MS_LOG(DEBUG) << "parse TfliteSquareParser";
    std::unique_ptr<schema::SquareT> attr(new schema::SquareT());
    op->primitive->value.type = schema::PrimitiveType_Square;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Sin") == 0) {
    MS_LOG(DEBUG) << "parse TfliteSinParser";
    std::unique_ptr<schema::SinT> attr(new schema::SinT());
    op->primitive->value.type = schema::PrimitiveType_Sin;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Cos") == 0) {
    MS_LOG(DEBUG) << "parse TfliteCosParser";
    std::unique_ptr<schema::CosT> attr(new schema::CosT());
    op->primitive->value.type = schema::PrimitiveType_Cos;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Log") == 0) {
    MS_LOG(DEBUG) << "parse TfliteLogParser";
    std::unique_ptr<schema::LogT> attr(new schema::LogT());
    op->primitive->value.type = schema::PrimitiveType_Log;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Round") == 0) {
    MS_LOG(DEBUG) << "parse TfliteRoundParser";
    std::unique_ptr<schema::RoundT> attr(new schema::RoundT());
    op->primitive->value.type = schema::PrimitiveType_Round;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Ceil") == 0) {
    MS_LOG(DEBUG) << "parse TfliteCeilParser";
    std::unique_ptr<schema::CeilT> attr(new schema::CeilT());
    op->primitive->value.type = schema::PrimitiveType_Ceil;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "flOOR") == 0) {
    MS_LOG(DEBUG) << "parse TfliteFloorParser";
    std::unique_ptr<schema::FloorT> attr(new schema::FloorT());
    op->primitive->value.type = schema::PrimitiveType_Floor;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "wrong op type";
    return RET_ERROR;
  }
}

STATUS TfliteCompareOpParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
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
  if (std::strcmp(node_name, "Equal") == 0) {
    MS_LOG(DEBUG) << "parse TfliteEqualParser";
    std::unique_ptr<schema::EqualT> attr(new schema::EqualT());
    op->primitive->value.type = schema::PrimitiveType_Equal;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "NotEqual") == 0) {
    MS_LOG(DEBUG) << "parse TfliteNotEqualParser";
    std::unique_ptr<schema::NotEqualT> attr(new schema::NotEqualT());
    op->primitive->value.type = schema::PrimitiveType_NotEqual;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Greater") == 0) {
    MS_LOG(DEBUG) << "parse TfliteGreaterParser";
    std::unique_ptr<schema::GreaterT> attr(new schema::GreaterT());
    op->primitive->value.type = schema::PrimitiveType_Greater;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "GreaterEqual") == 0) {
    MS_LOG(DEBUG) << "parse TfliteGreaterEqualParser";
    std::unique_ptr<schema::GreaterEqualT> attr(new schema::GreaterEqualT());
    op->primitive->value.type = schema::PrimitiveType_GreaterEqual;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "Less") == 0) {
    MS_LOG(DEBUG) << "parse TfliteLessParser";
    std::unique_ptr<schema::LessT> attr(new schema::LessT());
    op->primitive->value.type = schema::PrimitiveType_Less;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else if (std::strcmp(node_name, "LessEqual") == 0) {
    MS_LOG(DEBUG) << "parse TfliteLessEqualParser";
    std::unique_ptr<schema::LessEqualT> attr(new schema::LessEqualT());
    op->primitive->value.type = schema::PrimitiveType_LessEqual;
    op->primitive->value.value = attr.release();
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "wrong op type";
    return RET_ERROR;
  }
}

TfliteNodeRegister g_tfliteAddParser("Add", new TfliteAddParser());
TfliteNodeRegister g_tfliteSubParser("Sub", new TfliteSubParser());
TfliteNodeRegister g_TfliteMulParser("Mul", new TfliteMulParser());
TfliteNodeRegister g_TfliteDivParser("Div", new TfliteDivParser());
TfliteNodeRegister g_tfliteFloorDivParser("FloorDiv", new TfliteFloorDivParser());
TfliteNodeRegister g_tfliteFloorModParser("FloorMod", new TfliteFloorModParser());
TfliteNodeRegister g_tfliteRealDivParser("RealDiv", new TfliteRealDivParser());
TfliteNodeRegister g_TflitePowParser("Pow", new TflitePowParser());
TfliteNodeRegister g_tfliteSquaredDifferenceParser("SquaredDifference", new TfliteSquaredDifferenceParser());
TfliteNodeRegister g_TfliteMaximumParser("Maximum", new TfliteMaximumParser());
TfliteNodeRegister g_TfliteMinimumParser("Minimum", new TfliteMinimumParser());

TfliteNodeRegister g_TfliteAbsParser("Abs", new TfliteAbsParser());
TfliteNodeRegister g_TfliteExpParser("Exp", new TfliteExpParser());
TfliteNodeRegister g_TfliteSqrtParser("Sqrt", new TfliteSqrtParser());
TfliteNodeRegister g_tfliteRsqrtParser("Rsqrt", new TfliteRsqrtParser());
TfliteNodeRegister g_TfliteSquareParser("Square", new TfliteSquareParser());
TfliteNodeRegister g_TfliteSinParser("Sin", new TfliteSinParser());
TfliteNodeRegister g_TfliteCosParser("Cos", new TfliteCosParser());
TfliteNodeRegister g_TfliteLogParser("Log", new TfliteLogParser());
TfliteNodeRegister g_tfliteRoundParser("Round", new TfliteRoundParser());
TfliteNodeRegister g_TfliteCeilParser("Ceil", new TfliteCeilParser());
TfliteNodeRegister g_tfliteFloorParser("flOOR", new TfliteFloorParser());

TfliteNodeRegister g_tfliteEqualParser("Equal", new TfliteEqualParser());
TfliteNodeRegister g_tfliteNotEqualParser("NotEqual", new TfliteNotEqualParser());
TfliteNodeRegister g_tfliteGreaterEParser("Greater", new TfliteGreaterParser());
TfliteNodeRegister g_tfliteGreaterEqualParser("GreaterEqual", new TfliteGreaterEqualParser());
TfliteNodeRegister g_tfliteLessParser("Less", new TfliteLessParser());
TfliteNodeRegister g_tfliteLessEqualParser("LessEqual", new TfliteLessEqualParser());

}  // namespace lite
}  // namespace mindspore


