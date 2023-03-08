/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/square_sum_fusion.h"

#include <memory>
#include <vector>
#include <tuple>
#include <string>

#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/core_ops.h"
#include "backend/common/optimizer/helper.h"
#include "include/backend/kernel_info.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
std::tuple<CNodePtr, AnfNodePtr, CNodePtr> GetPrevNodes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto sum = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sum);
  CheckCNodeInputSize(sum, kSumNodeInputTensorNum);
  auto square_anf = sum->input(1);
  MS_EXCEPTION_IF_NULL(square_anf);
  auto square = square_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(square);

  return std::make_tuple(sum, square_anf, square);
}
}  // namespace

CNodePtr SquareSumFusion::GenerateSquareSumV1(const FuncGraphPtr &graph, const CNodePtr &square,
                                              const CNodePtr &sum) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(square);
  MS_EXCEPTION_IF_NULL(sum);
  CheckCNodeInputSize(square, kSquareNodeInputTensorNum);
  auto prim = std::make_shared<Primitive>(kSquareSumV1OpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> square_sumv1_inputs = {NewValueNode(prim), square->input(1)};
  auto square_sumv1 = NewCNode(square_sumv1_inputs, graph);
  MS_EXCEPTION_IF_NULL(square_sumv1);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  square_sumv1->set_kernel_info(kernel_info);
  auto types = {common::AnfAlgo::GetOutputInferDataType(sum, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(sum, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, square_sumv1.get());
  square_sumv1->set_scope(sum->scope());
  common::AnfAlgo::CopyNodeAttr(kAttrAxis, sum, square_sumv1);
  common::AnfAlgo::CopyNodeAttr(kAttrKeepDims, sum, square_sumv1);
  auto names = MakeValue<std::vector<std::string>>({square->fullname_with_scope(), sum->fullname_with_scope()});
  common::AnfAlgo::SetNodeAttr(kAttrDatadumpOriginalNames, names, square_sumv1);
  return square_sumv1;
}

CNodePtr SquareSumFusion::GenerateSquareSumV2(const FuncGraphPtr &graph, const CNodePtr &square,
                                              const CNodePtr &sum) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(square);
  MS_EXCEPTION_IF_NULL(sum);
  CheckCNodeInputSize(square, kSquareNodeInputTensorNum);
  auto prim = std::make_shared<Primitive>(kSquareSumV2OpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> square_sumv2_inputs = {NewValueNode(prim), square->input(1)};
  auto square_sumv2 = NewCNode(square_sumv2_inputs, graph);
  MS_EXCEPTION_IF_NULL(square_sumv2);
  auto types = {common::AnfAlgo::GetOutputInferDataType(sum, 0), common::AnfAlgo::GetOutputInferDataType(square, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(sum, 0), AnfAlgo::GetOutputDetailShape(square, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, square_sumv2.get());
  square_sumv2->set_scope(sum->scope());
  common::AnfAlgo::CopyNodeAttr(kAttrAxis, sum, square_sumv2);
  common::AnfAlgo::CopyNodeAttr(kAttrKeepDims, sum, square_sumv2);
  auto names = MakeValue<std::vector<std::string>>({square->fullname_with_scope(), sum->fullname_with_scope()});
  common::AnfAlgo::SetNodeAttr(kAttrDatadumpOriginalNames, names, square_sumv2);
  return square_sumv2;
}

const BaseRef SquareSumFusion::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(X);
  return VectorRef({prim::kPrimReduceSumD, VectorRef({prim::kPrimSquare, X})});
}

const AnfNodePtr SquareSumFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  CNodePtr sum = nullptr;
  AnfNodePtr square_anf = nullptr;
  CNodePtr square = nullptr;
  std::tie(sum, square_anf, square) = GetPrevNodes(node);
  constexpr size_t kShape2Dim = 2;

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(square_anf) == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "Square node has no output in NodeUsersMap" << trace::DumpSourceLines(node);
  }
  AnfNodePtr ret_node = nullptr;
  if (manager->node_users()[square_anf].size() == 1) {
    ret_node = GenerateSquareSumV1(graph, square, sum);
  } else if (manager->node_users()[square_anf].size() == kShape2Dim) {
    auto square_sumv2 = GenerateSquareSumV2(graph, square, sum);

    std::vector<AnfNodePtr> square_sumv2_outputs;
    CreateMultipleOutputsOfAnfNode(graph, square_sumv2, kSquareSumv2OutputNum, &square_sumv2_outputs);
    if (square_sumv2_outputs.size() != kSquareSumv2OutputNum) {
      MS_LOG(EXCEPTION) << "make SquareSumV2 outputs fail" << trace::DumpSourceLines(square_sumv2);
    }
    (void)manager->Replace(square, square_sumv2_outputs[1]);
    ret_node = square_sumv2_outputs[0];
  }
  return ret_node;
}
}  // namespace opt
}  // namespace mindspore
