/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "session/ascend_inference_session.h"
#include "operator/ops.h"
#include "ir/tensor.h"
#include "ir/anf.h"
#include "ir/param_value.h"
#include "device/kernel_runtime.h"
#include "session/anf_runtime_algorithm.h"
#include "common/utils.h"
#include "common/trans.h"
#include "kernel/tbe/tbe_python_funcs.h"
#include "utils/config_manager.h"
#include "utils/base_ref_extends.h"

namespace mindspore {
namespace session {
void AscendInferenceSession::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                                           const std::vector<tensor::TensorPtr> &inputs_const) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  auto input_nodes = kernel_graph->inputs();

  size_t no_weight_input = 0;
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    tensor::TensorPtr tensor = nullptr;
    if (!input_nodes[i]->isa<Parameter>()) {
      MS_LOG(ERROR) << "Kernel graph inputs have anfnode which is not Parameter";
      continue;
    }
    auto pk_node = input_nodes[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(pk_node);
    auto device_address = AnfAlgo::GetMutableOutputAddr(pk_node, 0);
    MS_EXCEPTION_IF_NULL(device_address);
    if (!AnfAlgo::IsParameterWeight(pk_node)) {
      tensor = inputs[no_weight_input++];
      if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(pk_node, 0),
                                            LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                            tensor->data_c())) {
        MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
      }
    }
  }
}

GraphId AscendInferenceSession::CompileGraph(NotNull<FuncGraphPtr> func_graph) {
  auto graph_id = AscendSession::CompileGraph(func_graph);
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // load weight data to device
  auto input_nodes = kernel_graph->inputs();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    if (!input_nodes[i]->isa<Parameter>()) {
      MS_LOG(ERROR) << "Kernel graph inputs have anfnode which is not Parameter";
      continue;
    }
    auto pk_node = input_nodes[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(pk_node);
    auto device_address = AnfAlgo::GetMutableOutputAddr(pk_node, 0);
    MS_EXCEPTION_IF_NULL(device_address);
    if (AnfAlgo::IsParameterWeight(pk_node)) {
      const auto &param_value = pk_node->default_param();
      MS_EXCEPTION_IF_NULL(param_value);
      auto tensor = std::dynamic_pointer_cast<tensor::Tensor>(param_value->value());
      MS_EXCEPTION_IF_NULL(tensor);
      if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(pk_node, 0),
                                            LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                            tensor->data_c())) {
        MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
      }
    }
  }
  return graph_id;
}
}  // namespace session
}  // namespace mindspore
