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

#include "plugin/device/ascend/optimizer/enhancer/skip_empty_tensor_output.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
// return output number if all outputs are empty tensor, otherwise return 0
size_t GetAllEmptyTensorOutputNum(const AnfNodePtr &node) {
  auto output_num = AnfUtils::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_num; ++i) {
    auto output_shape = common::AnfAlgo::GetOutputInferShape(node, i);
    if (std::none_of(output_shape.cbegin(), output_shape.cend(), [](int64_t dim) { return dim == 0; })) {
      return 0;
    }
  }
  return output_num;
}

AnfNodePtr CreateEmptyTensorValueNode(const KernelGraphPtr &graph, const TypeId &type_id,
                                      const ShapeVector &output_shape) {
  auto empty_tensor = std::make_shared<tensor::Tensor>(type_id, output_shape);
  return graph->NewValueNode(empty_tensor);
}

AnfNodePtr CreateEmptyTensorOutputNode(const KernelGraphPtr &graph, const AnfNodePtr &node, size_t output_num) {
  if (output_num == 1) {
    return CreateEmptyTensorValueNode(graph, common::AnfAlgo::GetOutputInferDataType(node, 0),
                                      common::AnfAlgo::GetOutputInferShape(node, 0));
  }
  std::vector<AnfNodePtr> empty_tensor_outputs;
  for (size_t i = 0; i < output_num; ++i) {
    (void)empty_tensor_outputs.emplace_back(CreateEmptyTensorValueNode(
      graph, common::AnfAlgo::GetOutputInferDataType(node, i), common::AnfAlgo::GetOutputInferShape(node, i)));
  }
  return CreateMakeTupleNode(graph, empty_tensor_outputs);
}
}  // namespace

const BaseRef SkipEmptyTensorOutputPass::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({X, Xs});
}

const AnfNodePtr SkipEmptyTensorOutputPass::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node) || common::AnfAlgo::IsDynamicShape(node)) {
    return nullptr;
  }
  auto empty_output_num = GetAllEmptyTensorOutputNum(node);
  if (empty_output_num == 0) {
    return nullptr;
  }

  MS_LOG(INFO) << "Match all empty tensor output node: " << node->fullname_with_scope();
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  return CreateEmptyTensorOutputNode(kernel_graph, node, empty_output_num);
}
}  // namespace opt
}  // namespace mindspore
