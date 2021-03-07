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
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/fusion/sub_fusion.h"
#include "ops/fusion/exp_fusion.h"
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

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxAddParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::AddFusion>();
  return prim.release();
}

ops::PrimitiveC *OnnxSubParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::SubFusion>();
  return prim.release();
}

ops::PrimitiveC *OnnxDivParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::DivFusion>();
  return prim.release();
}

ops::PrimitiveC *OnnxMulParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::MulFusion>();
  return prim.release();
}

ops::PrimitiveC *OnnxEqualParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Equal>();
  return prim.release();
}

ops::PrimitiveC *OnnxLessParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Less>();
  return prim.release();
}

ops::PrimitiveC *OnnxGreaterParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Greater>();
  return prim.release();
}

ops::PrimitiveC *OnnxFloorParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Floor>();
  return prim.release();
}

ops::PrimitiveC *OnnxAbsParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Abs>();
  return prim.release();
}

ops::PrimitiveC *OnnxExpParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::ExpFusion>();

  prim->set_base(-1.0);
  prim->set_scale(1.0);
  prim->set_shift(0.0);

  return prim.release();
}

ops::PrimitiveC *OnnxCosParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Cos>();
  return prim.release();
}

ops::PrimitiveC *OnnxCeilParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Ceil>();
  return prim.release();
}

ops::PrimitiveC *OnnxLogParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Log>();
  return prim.release();
}

ops::PrimitiveC *OnnxAtanParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Atan>();
  return prim.release();
}

ops::PrimitiveC *OnnxAsinParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Asin>();
  return prim.release();
}

ops::PrimitiveC *OnnxAndParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::LogicalAnd>();
  return prim.release();
}

ops::PrimitiveC *OnnxOrParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::LogicalOr>();
  return prim.release();
}

ops::PrimitiveC *OnnxNotParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::LogicalNot>();
  return prim.release();
}

ops::PrimitiveC *OnnxNegParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Neg>();
  return prim.release();
}

ops::PrimitiveC *OnnxRoundParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Round>();
  return prim.release();
}

ops::PrimitiveC *OnnxSinParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Sin>();
  return prim.release();
}

ops::PrimitiveC *OnnxTanParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Tan>();
  return prim.release();
}

ops::PrimitiveC *OnnxSqrtParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Sqrt>();
  return prim.release();
}

ops::PrimitiveC *OnnxPowParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::PowFusion>();

  prim->set_scale(1.0);
  prim->set_shift(0.0);

  return prim.release();
}

ops::PrimitiveC *OnnxMinParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Minimum>();
  return prim.release();
}

ops::PrimitiveC *OnnxMaxParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Maximum>();
  return prim.release();
}

ops::PrimitiveC *OnnxEltwiseParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Eltwise>();

  if (onnx_node.op_type() == "Sum") {
    prim->set_mode(mindspore::EltwiseMode::SUM);
  } else {
    MS_LOG(ERROR) << "unsupported Eltwise type";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *OnnxReciprocalParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Reciprocal>();
  return prim.release();
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
}  // namespace lite
}  // namespace mindspore
