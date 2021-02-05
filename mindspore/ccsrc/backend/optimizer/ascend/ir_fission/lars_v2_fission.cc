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
#include "backend/optimizer/ascend/ir_fission/lars_v2_fission.h"
#include <memory>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"
#include "utils/utils.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
void CreateOutputsOfSquareSumAll(const FuncGraphPtr &graph, const CNodePtr &lars_v2,
                                 std::vector<AnfNodePtr> *square_sum_all_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(lars_v2);
  CheckCNodeInputSize(lars_v2, kLarsV2InputTensorNum);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(kSquareSumAllOpName)), lars_v2->input(1),
                                    lars_v2->input(2)};
  auto square_sum_all = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(square_sum_all);
  square_sum_all->set_scope(lars_v2->scope());

  auto types = {kNumberTypeFloat32, kNumberTypeFloat32};
  std::vector<size_t> shape;
  auto shapes = {shape, shape};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, square_sum_all.get());

  CreateMultipleOutputsOfAnfNode(graph, square_sum_all, 2, square_sum_all_outputs);
}

CNodePtr CreateLarsV2Update(const FuncGraphPtr &graph, const CNodePtr &lars_v2,
                            const std::vector<AnfNodePtr> &square_sum_all_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(lars_v2);
  if (square_sum_all_outputs.size() != 2) {
    MS_LOG(EXCEPTION) << "square_sum_all_outputs' size not equal 2"
                      << " trace: " << trace::DumpSourceLines(lars_v2);
  }
  CheckCNodeInputSize(lars_v2, kLarsV2InputTensorNum);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(kLarsV2UpdateOpName)),
                                    lars_v2->input(1),
                                    lars_v2->input(2),
                                    square_sum_all_outputs[0],
                                    square_sum_all_outputs[1],
                                    lars_v2->input(3),
                                    lars_v2->input(4)};
  auto lars_v2_update = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(lars_v2_update);
  lars_v2_update->set_scope(lars_v2->scope());
  lars_v2_update->set_abstract(lars_v2->abstract());
  return lars_v2_update;
}
}  // namespace

const BaseRef LarsV2Fission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto lars_v2_prim = std::make_shared<Primitive>(kLarsV2OpName);
  return VectorRef({lars_v2_prim, Xs});
}

const AnfNodePtr LarsV2Fission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto lars_v2 = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(lars_v2);

  std::vector<AnfNodePtr> square_sum_all_outputs;
  CreateOutputsOfSquareSumAll(graph, lars_v2, &square_sum_all_outputs);
  return CreateLarsV2Update(graph, lars_v2, square_sum_all_outputs);
}
}  // namespace opt
}  // namespace mindspore
