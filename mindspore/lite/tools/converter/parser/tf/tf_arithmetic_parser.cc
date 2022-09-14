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
#include "ops/squared_difference.h"
#include "ops/rsqrt.h"
#include "ops/round.h"
#include "ops/ceil.h"
#include "ops/fusion/exp_fusion.h"
#include "ops/floor.h"
#include "ops/floor_div.h"
#include "ops/floor_mod.h"
#include "ops/log.h"
#include "ops/sqrt.h"
#include "ops/cos.h"
#include "ops/sin.h"
#include "ops/square.h"
#include "ops/fusion/pow_fusion.h"
#include "ops/abs.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFAddParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::AddFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFSubParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::SubFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFMulParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::MulFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFDivParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::DivFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }
  std::string original_name = tf_op.op();
  (void)prim_c->AddAttr(ops::kOriginalOpName, MakeValue(original_name));
  return prim->GetPrim();
}

PrimitiveCPtr TFMaximumParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Maximum>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFMinimumParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Minimum>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFGreaterParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Greater>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFGreaterEqualParser::Parse(const tensorflow::NodeDef &tf_op,
                                          const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                          std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::GreaterEqual>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFLessParser::Parse(const tensorflow::NodeDef &tf_op,
                                  const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                  std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Less>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFLessEqualParser::Parse(const tensorflow::NodeDef &tf_op,
                                       const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                       std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::LessEqual>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFEqualParser::Parse(const tensorflow::NodeDef &tf_op,
                                   const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                   std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Equal>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFNotEqualParser::Parse(const tensorflow::NodeDef &tf_op,
                                      const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                      std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::NotEqual>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFSquaredDifferenceParser::Parse(const tensorflow::NodeDef &tf_op,
                                               const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                               std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::SquaredDifference>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFRsqrtParser::Parse(const tensorflow::NodeDef &tf_op,
                                   const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                   std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Rsqrt>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFRoundParser::Parse(const tensorflow::NodeDef &tf_op,
                                   const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                   std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Round>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFCeilParser::Parse(const tensorflow::NodeDef &tf_op,
                                  const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                  std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Ceil>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFExpParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::ExpFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFFloorParser::Parse(const tensorflow::NodeDef &tf_op,
                                   const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                   std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Floor>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFFloorDivParser::Parse(const tensorflow::NodeDef &tf_op,
                                      const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                      std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::FloorDiv>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFFloorModParser::Parse(const tensorflow::NodeDef &tf_op,
                                      const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                      std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::FloorMod>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFLogParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Log>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFSqrtParser::Parse(const tensorflow::NodeDef &tf_op,
                                  const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                  std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Sqrt>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFCosParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Cos>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFSinParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Sin>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFSquareParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Square>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFPowParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::PowFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}

PrimitiveCPtr TFAbsParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Abs>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}

TFNodeRegistrar g_tfAddParser("Add", new TFAddParser());
TFNodeRegistrar g_tfAddV2Parser("AddV2", new TFAddParser());
TFNodeRegistrar g_tfSubParser("Sub", new TFSubParser());
TFNodeRegistrar g_tfMulParser("Mul", new TFMulParser());
TFNodeRegistrar g_tfDivParser("Div", new TFDivParser());
TFNodeRegistrar g_tfRealDivParser("RealDiv", new TFDivParser());
TFNodeRegistrar g_tfMaximumParser("Maximum", new TFMaximumParser());
TFNodeRegistrar g_tfMinimumParser("Minimum", new TFMinimumParser());
TFNodeRegistrar g_tfGreaterParser("Greater", new TFGreaterParser());
TFNodeRegistrar g_tfGreaterEqualParser("GreaterEqual", new TFGreaterEqualParser());
TFNodeRegistrar g_tfLessParser("Less", new TFLessParser());
TFNodeRegistrar g_tfLessEqualParser("LessEqual", new TFLessEqualParser());
TFNodeRegistrar g_tfEqualParser("Equal", new TFEqualParser());
TFNodeRegistrar g_tfNotEqualParser("NotEqual", new TFNotEqualParser());
TFNodeRegistrar g_tfSquaredDifferenceParser("SquaredDifference", new TFSquaredDifferenceParser());
TFNodeRegistrar g_tfRsqrtParser("Rsqrt", new TFRsqrtParser());

TFNodeRegistrar g_tfRoundParser("Round", new TFRoundParser());
TFNodeRegistrar g_tfCosParser("Cos", new TFCosParser());
TFNodeRegistrar g_tfSinParser("Sin", new TFSinParser());
TFNodeRegistrar g_tfSquareParser("Square", new TFSquareParser());
TFNodeRegistrar g_tfCeilParser("Ceil", new TFCeilParser());
TFNodeRegistrar g_tfExpParser("Exp", new TFExpParser());
TFNodeRegistrar g_tfFloorParser("Floor", new TFFloorParser());
TFNodeRegistrar g_tfFloorDivParser("FloorDiv", new TFFloorDivParser());
TFNodeRegistrar g_tfFloorModParser("FloorMod", new TFFloorModParser());
TFNodeRegistrar g_tfLogParser("Log", new TFLogParser());
TFNodeRegistrar g_tfSqrtParser("Sqrt", new TFSqrtParser());
TFNodeRegistrar g_tfPowParser("Pow", new TFPowParser());
TFNodeRegistrar g_tfAbsParser("Abs", new TFAbsParser());
}  // namespace lite
}  // namespace mindspore
