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

#include "src/train/lite_kernel_runtime.h"
#include "backend/session/anf_runtime_algorithm.h"
namespace mindspore::lite {
std::vector<CNodePtr> LiteInferKernelRuntime::GetGraphInputs(const std::vector<CNodePtr> &execution_order) {
  std::vector<CNodePtr> graph_inputs;
  for (const auto &cnode : execution_order) {
    bool is_graph_inputs = true;
    for (const auto &input : cnode->inputs()) {
      if (input->isa<CNode>()) {
        is_graph_inputs = false;
        break;
      }
    }
    if (is_graph_inputs) {
      graph_inputs.emplace_back(cnode);
    }
  }
  return graph_inputs;
}

void LiteInferKernelRuntime::BindInputOutput(const session::KernelGraph *graph,
                                             const std::vector<tensor::Tensor *> &inputs,
                                             std::vector<tensor::Tensor *> *outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  auto execution_order = graph->execution_order();
  auto graph_inputs = GetGraphInputs(execution_order);
  int input_count = 0;
  for (const auto &graph_input : graph_inputs) {
    auto liteKernel = dynamic_cast<kernel::LiteKernel *>(AnfAlgo::GetKernelMod(graph_input));
    for (auto input_tensor : liteKernel->GetInputs()) {
      if (schema::NodeType_ValueNode == input_tensor->TensorType() && input_tensor->Data() != nullptr) {
        continue;
      }
      input_tensor->SetData(inputs[input_count]->Data());
      input_count++;
    }
  }

  auto return_node = graph->get_return();
  for (const auto &return_input : return_node->inputs()) {
    if (return_input->isa<CNode>()) {
      auto liteKernel = dynamic_cast<kernel::LiteKernel *>(AnfAlgo::GetKernelMod(return_input));
      auto output_tensors = liteKernel->GetOutputs();
      for (auto output_tensor : output_tensors) {
        // tensor::TensorPtr output_tensor_ptr(output_tensor);
        outputs->push_back(output_tensor);
      }
    }
  }
}

bool LiteInferKernelRuntime::Run(session::KernelGraph *graph, const std::vector<tensor::Tensor *> &inputs,
                                 std::vector<tensor::Tensor *> *outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  BindInputOutput(graph, inputs, *outputs);
  std::vector<kernel::LiteKernel *> kernels;
  auto nodes = graph->execution_order();
  for (const auto &node : nodes) {
    auto liteKernel = dynamic_cast<kernel::LiteKernel *>(AnfAlgo::GetKernelMod(node));
    if (liteKernel == nullptr) {
      continue;
    }
    kernels.emplace_back(liteKernel);
  }
  kernel::LiteKernelUtil::TopologicalSortKernels(kernels);
  Executor executor;
  auto ret = executor.Run(inputs, *outputs, kernels);
  return 0 == ret;
}
}  // namespace mindspore::lite
