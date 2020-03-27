/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "device/cpu/cpu_session.h"
#include <algorithm>
#include "ir/meta_tensor.h"
#include "ir/anf.h"
#include "kernel/kernel.h"
#include "common/utils.h"
#include "session/anf_runtime_algorithm.h"
#include "device/kernel_runtime.h"
#include "predict/predict.h"
#include "device/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace session {
GraphId CPUSession::CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  auto graph_id = graph_sum_;
  auto graph = ConstructKernelGraph(lst, outputs);
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "set kernel info";
  SetKernelInfo(graph.get());
  predictmodel::StepConvertGraph(graph);
  MS_LOG(INFO) << "build kernel";
  BuildKernel(graph.get());
  MS_LOG(INFO) << "assign kernel address";
  runtime_.AssignKernelAddress(graph.get());
  return graph_id;
}

void CPUSession::RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  auto &kernel_graph = graphs_[graph_id];
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "bind input output address";
  runtime_.BindInputOutput(kernel_graph.get(), inputs, outputs);
  MS_LOG(INFO) << "run graph start";
  predictmodel::StepConvertWeight(inputs);
  auto execution_order = kernel_graph->execution_order();
  Reorder(&execution_order);
  kernel_graph->set_execution_order(execution_order);
  bool ret = runtime_.Run(kernel_graph.get());
  if (!ret) {
    MS_LOG(EXCEPTION) << "run graph failed";
  }
  MS_LOG(INFO) << "run graph end";
}

void CPUSession::SetKernelInfo(const KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &kernel_nodes = kernel_graph->execution_order();
  for (const auto &kernel_node : kernel_nodes) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      auto input_kernel_node = kernel_node->input(input_index + 1);
      MS_EXCEPTION_IF_NULL(input_kernel_node);
      if (!input_kernel_node->isa<Parameter>()) {
        continue;
      }
      auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
      MS_EXCEPTION_IF_NULL(builder);
      std::vector<std::string> output_formats = {kOpFormat_DEFAULT};
      builder->SetOutputsFormat(output_formats);
      std::vector<TypeId> output_types{kNumberTypeFloat32};
      builder->SetOutputsDeviceType(output_types);
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), input_kernel_node.get());
    }

    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    MS_EXCEPTION_IF_NULL(builder);
    std::vector<std::string> input_formats;
    std::vector<TypeId> input_types;
    std::vector<std::string> output_formats;
    std::vector<TypeId> output_types;

    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      input_formats.emplace_back(kOpFormat_DEFAULT);
      input_types.emplace_back(kNumberTypeFloat32);
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    for (size_t output_index = 0; output_index < output_num; ++output_index) {
      output_formats.emplace_back(kOpFormat_DEFAULT);
      output_types.emplace_back(kNumberTypeFloat32);
    }
    builder->SetInputsFormat(input_formats);
    builder->SetInputsDeviceType(input_types);
    builder->SetOutputsFormat(output_formats);
    builder->SetOutputsDeviceType(output_types);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel_node.get());
  }
}

void CPUSession::BuildKernel(const KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &kernel_nodes = kernel_graph->execution_order();
  for (const auto &kernel_node : kernel_nodes) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    MS_LOG(INFO) << "Cpu building operator[" << kernel_name << "].";
    std::shared_ptr<device::cpu::CPUKernel> cpu_kernel = device::cpu::CPUKernelFactory::Get().Create(kernel_name);
    if (cpu_kernel == nullptr) {
      MS_LOG(EXCEPTION) << "Operator[" << kernel_name << "] is not support.";
    }
    cpu_kernel->Init(kernel_node);
    AnfAlgo::SetKernelMod(cpu_kernel, kernel_node.get());
    MS_LOG(INFO) << "Cpu build success operator[" << kernel_name << "].";
  }
}
}  // namespace session
}  // namespace mindspore
