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
ops::PrimitiveC *TFAddParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::AddFusion>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFSubParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::SubFusion>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFMulParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::MulFusion>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFDivParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::DivFusion>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFMaximumParser::Parse(const tensorflow::NodeDef &tf_op,
                                        const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                        std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Maximum>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFMinimumParser::Parse(const tensorflow::NodeDef &tf_op,
                                        const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                        std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Minimum>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFGreaterParser::Parse(const tensorflow::NodeDef &tf_op,
                                        const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                        std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Greater>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFGreaterEqualParser::Parse(const tensorflow::NodeDef &tf_op,
                                             const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                             std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::GreaterEqual>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFLessParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Less>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFLessEqualParser::Parse(const tensorflow::NodeDef &tf_op,
                                          const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                          std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::LessEqual>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFEqualParser::Parse(const tensorflow::NodeDef &tf_op,
                                      const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                      std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Equal>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFNotEqualParser::Parse(const tensorflow::NodeDef &tf_op,
                                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                         std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::NotEqual>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFSquaredDifferenceParser::Parse(const tensorflow::NodeDef &tf_op,
                                                  const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                                  std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::SquaredDifference>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFRsqrtParser::Parse(const tensorflow::NodeDef &tf_op,
                                      const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                      std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Rsqrt>();

  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }

  return prim.release();
}

ops::PrimitiveC *TFRoundParser::Parse(const tensorflow::NodeDef &tf_op,
                                      const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                      std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Round>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFCeilParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Ceil>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFExpParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::ExpFusion>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFFloorParser::Parse(const tensorflow::NodeDef &tf_op,
                                      const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                      std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Floor>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFFloorDivParser::Parse(const tensorflow::NodeDef &tf_op,
                                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                         std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::FloorDiv>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFFloorModParser::Parse(const tensorflow::NodeDef &tf_op,
                                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                         std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::FloorMod>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFLogParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Log>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFSqrtParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Sqrt>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFCosParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Cos>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFSinParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Sin>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFSquareParser::Parse(const tensorflow::NodeDef &tf_op,
                                       const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                       std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Square>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFPowParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::PowFusion>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
}

ops::PrimitiveC *TFAbsParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Abs>();

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
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
