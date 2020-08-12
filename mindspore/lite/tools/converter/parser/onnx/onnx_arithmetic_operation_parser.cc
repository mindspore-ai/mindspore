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

#include <memory>
#include "tools/converter/parser/onnx/onnx_arithmetic_operation_parser.h"

namespace mindspore {
namespace lite {
STATUS OnnxAddParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx AddParser";
  if (op != nullptr) {
    std::unique_ptr<schema::AddT> attr(new schema::AddT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Add;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

STATUS OnnxSubParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx SubParser";
  if (op != nullptr) {
    std::unique_ptr<schema::SubT> attr(new schema::SubT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Sub;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

STATUS OnnxMulParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx MulParser";
  if (op != nullptr) {
    std::unique_ptr<schema::MulT> attr(new schema::MulT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Mul;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

STATUS OnnxDivParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx DivParser";
  if (op != nullptr) {
    std::unique_ptr<schema::DivT> attr(new schema::DivT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Div;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

STATUS OnnxPowParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx PowParser";
  if (op != nullptr) {
    // TODO(wangzhe) attr power need populate
    std::unique_ptr<schema::PowerT> attr(new schema::PowerT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Power;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxEqualParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                              schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx EqualParser";
  if (op != nullptr) {
    std::unique_ptr<schema::EqualT> attr(new schema::EqualT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Equal;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

STATUS OnnxLessParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx LessParser";
  if (op != nullptr) {
    std::unique_ptr<schema::LessT> attr(new schema::LessT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Less;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxGreaterParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx GreaterParser";
  if (op != nullptr) {
    std::unique_ptr<schema::GreaterT> attr(new schema::GreaterT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Greater;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

STATUS OnnxMinParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx MinParser";
  if (op != nullptr) {
    std::unique_ptr<schema::MinT> attr(new schema::MinT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Min;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

STATUS OnnxEltwiseParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx EltwiseParser";
  std::unique_ptr<schema::EltwiseT> attr(new schema::EltwiseT());
  // there is no Prod in onnx
  if (onnx_node.op_type() == "Sum") {
    attr->mode = schema::EltwiseMode_SUM;
  } else if (onnx_node.op_type() == "Max") {
    attr->mode = schema::EltwiseMode_MAXIMUM;
  }

  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Eltwise;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

STATUS OnnxFloorParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                              schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx FloorParser";
  if (op != nullptr) {
    std::unique_ptr<schema::FloorT> attr(new schema::FloorT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Floor;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxAbsParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx AbsParser";
  if (op != nullptr) {
    std::unique_ptr<schema::AbsT> attr(new schema::AbsT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Abs;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxNegParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx NegParser";
  if (op != nullptr) {
    std::unique_ptr<schema::NegT> attr(new schema::NegT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Neg;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxExpParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx ExpParser";
  if (op != nullptr) {
    std::unique_ptr<schema::ExpT> attr(new schema::ExpT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Exp;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxCosParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx CosParser";
  if (op != nullptr) {
    std::unique_ptr<schema::CosT> attr(new schema::CosT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Cos;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxSinParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx SinParser";
  if (op != nullptr) {
    std::unique_ptr<schema::SinT> attr(new schema::SinT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Sin;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxSqrtParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx SqrtParser";
  if (op != nullptr) {
    std::unique_ptr<schema::SqrtT> attr(new schema::SqrtT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Sqrt;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxCeilParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx CeilParser";
  if (op != nullptr) {
    std::unique_ptr<schema::CeilT> attr(new schema::CeilT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Ceil;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxLogParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx LogParser";
  if (op != nullptr) {
    std::unique_ptr<schema::LogT> attr(new schema::LogT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Log;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxTanParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx TanParser";
  if (op != nullptr) {
    std::unique_ptr<schema::TanT> attr(new schema::TanT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Tan;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxAtanParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx AtanParser";
  if (op != nullptr) {
    std::unique_ptr<schema::AtanT> attr(new schema::AtanT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Atan;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}
STATUS OnnxAsinParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx AsinParser";
  if (op != nullptr) {
    std::unique_ptr<schema::AsinT> attr(new schema::AsinT());
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Asin;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

STATUS OnnxTanhParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx TanhParser";
  if (op != nullptr) {
    MS_LOG(ERROR) << "mslite don't support tanh now";
    return RET_ERROR;
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxAddParser("Add", new OnnxAddParser());
OnnxNodeRegistrar g_onnxInt8AddParser("Int8Add", new OnnxAddParser());
OnnxNodeRegistrar g_onnxSubParser("Sub", new OnnxSubParser());
OnnxNodeRegistrar g_onnxMulParser("Mul", new OnnxMulParser());
OnnxNodeRegistrar g_onnxDivParser("Div", new OnnxDivParser());
// OnnxNodeRegistrar g_onnxMeanParser("Mean", new OnnxMeanParser());  // onnx's Mean is different from mslite's
OnnxNodeRegistrar g_onnxPowParser("Power", new OnnxPowParser());
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
}  // namespace lite
}  // namespace mindspore
