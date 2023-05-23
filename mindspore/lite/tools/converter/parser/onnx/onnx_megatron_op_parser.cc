/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_megatron_op_parser.h"
#include <memory>
#include <vector>
#include "ops/fusion/layer_norm_fusion.h"
#include "tools/converter/ops/ops_def.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kParallel = "parallel";
constexpr auto kParallelNum = "parallel_num";
}  // namespace

PrimitiveCPtr OnnxMegatronAllReduceParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<lite::MegatronAllReduce>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == kParallelNum) {
      prim->AddAttr(kParallelNum, MakeValue(onnx_node_attr.i()));
    }
  }
  return prim;
}

PrimitiveCPtr OnnxMegatronLinearAllGatherParser::Parse(const onnx::GraphProto &onnx_graph,
                                                       const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<lite::MegatronLinearAllGather>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == kParallel) {
      bool parallel = onnx_node_attr.i() == 1;
      prim->AddAttr(kParallel, MakeValue(parallel));
    }
  }
  return prim;
}

PrimitiveCPtr OnnxMegatronMakeViewlessTensorParser::Parse(const onnx::GraphProto &onnx_graph,
                                                          const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<lite::MegatronMakeViewlessTensor>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim;
}

PrimitiveCPtr OnnxMegatronScaledMaskedSoftmaxParser::Parse(const onnx::GraphProto &onnx_graph,
                                                           const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<lite::MegatronScaledMaskedSoftmax>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "scale") {
      prim->AddAttr(kAttrScales, MakeValue(onnx_node_attr.f()));
    }
  }
  return prim;
}

PrimitiveCPtr OnnxMegatronFusedLayerNormAffineParser::Parse(const onnx::GraphProto &onnx_graph,
                                                            const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::LayerNormFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "eps") {
      prim->set_epsilon(onnx_node_attr.i());
    }
    prim->set_begin_norm_axis(-1);
    prim->set_begin_params_axis(-1);
  }
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxMegatronMakeViewlessTensorParser("MakeViewlessTensor",
                                                         new OnnxMegatronMakeViewlessTensorParser());
OnnxNodeRegistrar g_onnxMegatronAllReduceParser("_ReduceFromModelParallelRegion", new OnnxMegatronAllReduceParser());
OnnxNodeRegistrar g_onnxMegatronLinearAllGatherParser("LinearWithGradAccumulationAndAsyncCommunication",
                                                      new OnnxMegatronLinearAllGatherParser());
OnnxNodeRegistrar g_onnxMegatronScaledMaskedSoftmaxParser("ScaledMaskedSoftmax",
                                                          new OnnxMegatronScaledMaskedSoftmaxParser());
OnnxNodeRegistrar g_onnxMegatronFusedLayerNormAffineParser("FusedLayerNormAffineFunction",
                                                           new OnnxMegatronFusedLayerNormAffineParser());
}  // namespace lite
}  // namespace mindspore
