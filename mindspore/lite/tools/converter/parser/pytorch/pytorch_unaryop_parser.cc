/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/pytorch/pytorch_unaryop_parser.h"
#include <memory>
#include "ops/fusion/tile_fusion.h"
#include "ops/identity.h"
#include "ops/erf.h"
#include "ops/fusion/exp_fusion.h"
#include "ops/floor.h"
#include "ops/abs.h"
#include "ops/cos.h"
#include "ops/ceil.h"
#include "ops/log.h"
#include "ops/atan.h"
#include "ops/asin.h"
#include "ops/neg.h"
#include "ops/round.h"
#include "ops/tan.h"
#include "ops/sqrt.h"
#include "ops/minimum.h"
#include "ops/maximum.h"
#include "ops/eltwise.h"
#include "ops/sin.h"
#include "ops/reciprocal.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
template <typename OPTy>
PrimitiveCPtr PytorchUnaryOpParser<OPTy>::Parse(const torch::jit::Node *torch_node,
                                                std::vector<size_t> *input_indices) {
  MS_ASSERT(torch_node != nullptr && input_indices != nullptr);
  auto prim = std::make_unique<OPTy>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  input_indices->push_back(0);
  return prim->GetPrim();
}

PytorchNodeRegistrar g_pytorchTileParser("tile", new PytorchUnaryOpParser<ops::TileFusion>());
PytorchNodeRegistrar g_pytorchIdentityParser("Identity", new PytorchUnaryOpParser<ops::Identity>());
PytorchNodeRegistrar g_pytorchErfParser("erf", new PytorchUnaryOpParser<ops::Erf>());
PytorchNodeRegistrar g_pytorchFloorParser("floor", new PytorchUnaryOpParser<ops::Floor>());
PytorchNodeRegistrar g_pytorchAbsParser("abs", new PytorchUnaryOpParser<ops::Abs>());
PytorchNodeRegistrar g_pytorchExpParser("exp", new PytorchUnaryOpParser<ops::ExpFusion>());
PytorchNodeRegistrar g_pytorchCosParser("cos", new PytorchUnaryOpParser<ops::Cos>());
PytorchNodeRegistrar g_pytorchCeilParser("ceil", new PytorchUnaryOpParser<ops::Ceil>());
PytorchNodeRegistrar g_pytorchLogParser("log", new PytorchUnaryOpParser<ops::Log>());
PytorchNodeRegistrar g_pytorchAtanParser("atan", new PytorchUnaryOpParser<ops::Atan>());
PytorchNodeRegistrar g_pytorchAsinParser("asin", new PytorchUnaryOpParser<ops::Asin>());
PytorchNodeRegistrar g_pytorchNegParser("neg", new PytorchUnaryOpParser<ops::Neg>());
PytorchNodeRegistrar g_pytorchRoundParser("round", new PytorchUnaryOpParser<ops::Round>());
PytorchNodeRegistrar g_pytorchSqrtParser("sqrt", new PytorchUnaryOpParser<ops::Sqrt>());
PytorchNodeRegistrar g_pytorchTanParser("tan", new PytorchUnaryOpParser<ops::Tan>());
PytorchNodeRegistrar g_pytorchSinParser("sin", new PytorchUnaryOpParser<ops::Sin>());
PytorchNodeRegistrar g_pytorchMinParser("min", new PytorchUnaryOpParser<ops::Minimum>());
PytorchNodeRegistrar g_pytorchMaxParser("max", new PytorchUnaryOpParser<ops::Maximum>());
PytorchNodeRegistrar g_pytorchReciprocalParser("reciprocal", new PytorchUnaryOpParser<ops::Reciprocal>());
}  // namespace lite
}  // namespace mindspore
