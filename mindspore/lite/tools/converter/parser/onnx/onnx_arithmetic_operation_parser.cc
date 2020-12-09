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

#include "tools/converter/parser/onnx/onnx_arithmetic_operation_parser.h"
#include <memory>
#include <numeric>
#include <functional>

namespace mindspore {
namespace lite {
lite::PrimitiveC *OnnxAddParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx AddParser";
  auto attr = std::make_unique<schema::AddT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Add;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxSubParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx SubParser";
  auto attr = std::make_unique<schema::SubT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Sub;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxMulParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx MulParser";
  auto attr = std::make_unique<schema::MulT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Mul;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxDivParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx DivParser";
  auto attr = std::make_unique<schema::DivT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Div;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxPowParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx PowParser";
  auto attr = std::make_unique<schema::PowerT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  attr->scale = 1.0f;
  attr->shift = 0.0f;
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Power;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxEqualParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                      const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx EqualParser";
  auto attr = std::make_unique<schema::EqualT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Equal;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxLessParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                     const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx LessParser";
  auto attr = std::make_unique<schema::LessT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Less;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxGreaterParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                        const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx GreaterParser";
  auto attr = std::make_unique<schema::GreaterT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Greater;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxMinParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx MinParser";
  auto attr = std::make_unique<schema::MinimumT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Minimum;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxEltwiseParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                        const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx EltwiseParser";
  auto attr = std::make_unique<schema::EltwiseT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  if (onnx_node.op_type() == "Sum") {
    attr->mode = schema::EltwiseMode_SUM;
  } else if (onnx_node.op_type() == "Max") {
    attr->mode = schema::EltwiseMode_MAXIMUM;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Eltwise;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxFloorParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                      const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx FloorParser";
  auto attr = std::make_unique<schema::FloorT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Floor;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxAbsParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx AbsParser";
  auto attr = std::make_unique<schema::AbsT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Abs;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxNegParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx NegParser";
  auto attr = std::make_unique<schema::NegT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Neg;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxExpParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx ExpParser";
  auto attr = std::make_unique<schema::ExpT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Exp;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxCosParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx CosParser";
  auto attr = std::make_unique<schema::CosT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Cos;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxSinParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx SinParser";
  auto attr = std::make_unique<schema::SinT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Sin;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxSqrtParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                     const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx SqrtParser";
  auto attr = std::make_unique<schema::SqrtT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Sqrt;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxCeilParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                     const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx CeilParser";
  auto attr = std::make_unique<schema::CeilT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Ceil;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxLogParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx LogParser";
  auto attr = std::make_unique<schema::LogT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Log;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxTanParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx TanParser";
  auto attr = std::make_unique<schema::TanT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Tan;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxAtanParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                     const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx AtanParser";
  auto attr = std::make_unique<schema::AtanT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Atan;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxAsinParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                     const onnx::NodeProto &onnx_node) {
  auto attr = std::make_unique<schema::AsinT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Asin;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxTanhParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                     const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx TanhParser";
  auto attr = std::make_unique<schema::ActivationT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  attr->type = schema::ActivationType_TANH;
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Activation;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxSignParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                     const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx TanhParser";
  auto attr = std::make_unique<schema::ActivationT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  attr->type = schema::ActivationType_SIGN;
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Activation;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxAndParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx AndParser";
  auto attr = std::make_unique<schema::LogicalAndT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_LogicalAnd;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxOrParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                   const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx OrParser";
  auto attr = std::make_unique<schema::LogicalOrT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_LogicalOr;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxNotParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                    const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx NotParser";
  auto attr = std::make_unique<schema::LogicalNotT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_LogicalNot;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxRoundParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                      const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx RoundParser";
  auto attr = std::make_unique<schema::RoundT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Round;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

lite::PrimitiveC *OnnxReciprocalParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                           const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx ReciprocalParser";
  auto attr = std::make_unique<schema::ReciprocalT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Reciprocal;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}
OnnxNodeRegistrar g_onnxAddParser("Add", new OnnxAddParser());
OnnxNodeRegistrar g_onnxInt8AddParser("Int8Add", new OnnxAddParser());
OnnxNodeRegistrar g_onnxSubParser("Sub", new OnnxSubParser());
OnnxNodeRegistrar g_onnxMulParser("Mul", new OnnxMulParser());
OnnxNodeRegistrar g_onnxDivParser("Div", new OnnxDivParser());
OnnxNodeRegistrar g_onnxPowParser("Pow", new OnnxPowParser());
OnnxNodeRegistrar g_onnxEqualParser("Equal", new OnnxEqualParser());
OnnxNodeRegistrar g_onnxLessParser("Less", new OnnxLessParser());
OnnxNodeRegistrar g_onnxGreaterParser("Greater", new OnnxGreaterParser());
OnnxNodeRegistrar g_onnxMinParser("Min", new OnnxMinParser());
OnnxNodeRegistrar g_onnxSumParser("Sum", new OnnxEltwiseParser());
OnnxNodeRegistrar g_onnxMaxParser("Max", new OnnxEltwiseParser());
OnnxNodeRegistrar g_onnxFloorParser("Floor", new OnnxFloorParser());
OnnxNodeRegistrar g_onnxAbsParser("Abs", new OnnxAbsParser());
OnnxNodeRegistrar g_onnxNegParser("Neg", new OnnxNegParser());
OnnxNodeRegistrar g_onnxExpParser("Exp", new OnnxExpParser());
OnnxNodeRegistrar g_onnxCosParser("Cos", new OnnxCosParser());
OnnxNodeRegistrar g_onnxSinParser("Sin", new OnnxSinParser());
OnnxNodeRegistrar g_onnxSqrtParser("Sqrt", new OnnxSqrtParser());
OnnxNodeRegistrar g_onnxCeilParser("Ceil", new OnnxCeilParser());
OnnxNodeRegistrar g_onnxLogParser("Log", new OnnxLogParser());
OnnxNodeRegistrar g_onnxTanParser("Tan", new OnnxTanParser());
OnnxNodeRegistrar g_onnxAtanParser("Atan", new OnnxAtanParser());
OnnxNodeRegistrar g_onnxAsinParser("Asin", new OnnxAsinParser());
OnnxNodeRegistrar g_onnxTanhParser("Tanh", new OnnxTanhParser());
OnnxNodeRegistrar g_onnxSignParser("Sign", new OnnxTanhParser());
OnnxNodeRegistrar g_onnxAndParser("And", new OnnxAndParser());
OnnxNodeRegistrar g_onnxOrParser("Or", new OnnxOrParser());
OnnxNodeRegistrar g_onnxNotParser("Not", new OnnxNotParser());
OnnxNodeRegistrar g_onnxRoundParser("Round", new OnnxRoundParser());
OnnxNodeRegistrar g_onnxReciprocalParser("Reciprocal", new OnnxReciprocalParser());
}  // namespace lite
}  // namespace mindspore
