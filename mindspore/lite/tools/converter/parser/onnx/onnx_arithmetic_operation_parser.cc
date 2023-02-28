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
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/fusion/sub_fusion.h"
#include "ops/fusion/exp_fusion.h"
#include "ops/real_div.h"
#include "ops/equal.h"
#include "ops/less.h"
#include "ops/greater.h"
#include "ops/floor.h"
#include "ops/abs.h"
#include "ops/cos.h"
#include "ops/ceil.h"
#include "ops/log.h"
#include "ops/atan.h"
#include "ops/asin.h"
#include "ops/logical_and.h"
#include "ops/logical_not.h"
#include "ops/logical_or.h"
#include "ops/neg.h"
#include "ops/round.h"
#include "ops/tan.h"
#include "ops/sqrt.h"
#include "ops/fusion/pow_fusion.h"
#include "ops/minimum.h"
#include "ops/maximum.h"
#include "ops/eltwise.h"
#include "ops/sin.h"
#include "ops/reciprocal.h"
#include "ops/mod.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxAddParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::AddFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxSubParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::SubFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxDivParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::DivFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->AddAttr(ops::kOriginalOpName, api::MakeValue(std::string(ops::kNameRealDiv)));
  return prim->GetPrim();
}

PrimitiveCPtr OnnxMulParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::MulFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxEqualParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Equal>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxLessParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Less>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxGreaterParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Greater>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxFloorParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Floor>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxAbsParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Abs>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxExpParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::ExpFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  prim->set_base(-1.0);
  prim->set_scale(1.0);
  prim->set_shift(0.0);

  return prim->GetPrim();
}

PrimitiveCPtr OnnxCosParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Cos>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxCeilParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Ceil>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxLogParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Log>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxAtanParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Atan>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxAsinParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Asin>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxAndParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::LogicalAnd>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxOrParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::LogicalOr>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxNotParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::LogicalNot>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxNegParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Neg>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxRoundParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Round>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxSinParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Sin>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxTanParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Tan>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxSqrtParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Sqrt>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxPowParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::PowFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  prim->set_scale(1.0);
  prim->set_shift(0.0);

  return prim->GetPrim();
}

PrimitiveCPtr OnnxMinParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Minimum>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxMaxParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Maximum>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxEltwiseParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Eltwise>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  if (onnx_node.op_type() == "Sum") {
    prim->set_mode(mindspore::EltwiseMode::SUM);
  } else {
    MS_LOG(ERROR) << "unsupported Eltwise type";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr OnnxReciprocalParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Reciprocal>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxModParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Mod>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
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
OnnxNodeRegistrar g_onnxMaxParser("Max", new OnnxMaxParser());
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
OnnxNodeRegistrar g_onnxAndParser("And", new OnnxAndParser());
OnnxNodeRegistrar g_onnxOrParser("Or", new OnnxOrParser());
OnnxNodeRegistrar g_onnxNotParser("Not", new OnnxNotParser());
OnnxNodeRegistrar g_onnxRoundParser("Round", new OnnxRoundParser());
OnnxNodeRegistrar g_onnxReciprocalParser("Reciprocal", new OnnxReciprocalParser());
OnnxNodeRegistrar g_onnxModParser("Mod", new OnnxModParser());
}  // namespace lite
}  // namespace mindspore
