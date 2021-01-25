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
  auto primitive_c = new (std::nothrow) ops::AddFusion;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new AddFusion failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxSubParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::SubFusion;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new SubFusion failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxDivParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::DivFusion;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new DivFusion failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxMulParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::MulFusion;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new MulFusion failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxEqualParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Equal;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Equal failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxLessParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Less;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Less failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxGreaterParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Greater;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Greater failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxFloorParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Floor;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Floor failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxAbsParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Abs;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Abs failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxExpParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::ExpFusion;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new ExpFusion failed";
    return nullptr;
  }

  primitive_c->set_base(-1.0);
  primitive_c->set_scale(1.0);
  primitive_c->set_shift(0.0);

  return primitive_c;
}

ops::PrimitiveC *OnnxCosParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Cos;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Cos failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxCeilParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Ceil;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Ceil failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxLogParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Log;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Log failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxAtanParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Atan;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Atan failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxAsinParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Asin;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Asin failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxAndParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::LogicalAnd;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new LogicalAnd failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxOrParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::LogicalOr;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new LogicalOr failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxNotParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::LogicalNot;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new LogicalNot failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxNegParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Neg;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Neg failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxRoundParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Round;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Round failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxSinParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Sin;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new sin failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxTanParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Tan;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Tan failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxSqrtParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Sqrt;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Sqrt failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxPowParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::PowFusion;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new PowFusion failed";
    return nullptr;
  }

  primitive_c->set_scale(1.0);
  primitive_c->set_shift(0.0);

  return primitive_c;
}

ops::PrimitiveC *OnnxMinParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Minimum;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Minimum failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxMaxParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Maximum;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Maximum failed";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxEltwiseParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Eltwise;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Eltwise failed";
    return nullptr;
  }

  if (onnx_node.op_type() == "Sum") {
    primitive_c->set_mode(mindspore::EltwiseMode::SUM);
  } else {
    MS_LOG(ERROR) << "unsupported Eltwise type";
    return nullptr;
  }

  return primitive_c;
}

ops::PrimitiveC *OnnxReciprocalParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto primitive_c = new (std::nothrow) ops::Reciprocal;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Reciprocal failed";
    return nullptr;
  }

  return primitive_c;
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
