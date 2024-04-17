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
#ifndef MINDSPORE_CCSRC_CXX_API_GRAPH_GRAPH_IMPL_H
#define MINDSPORE_CCSRC_CXX_API_GRAPH_GRAPH_IMPL_H
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "include/api/cell.h"
#include "include/api/graph.h"
#include "cxx_api/graph/graph_data.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/base/base_ref_utils.h"
#include "include/backend/kernel_graph.h"
#include "backend/common/session/session_basic.h"
#include "backend/graph_compiler/backend.h"

namespace mindspore {
class GraphCell::GraphImpl {
 public:
  GraphImpl()
      : graph_(nullptr),
        graph_context_(nullptr),
        backend_(nullptr),
        actor_info_(""),
        kernel_graph_(),
        device_id_(0),
        inputs_info_(),
        outputs_info_(),
        input_names_(),
        output_names_(),
        load_flag_(false) {}
  virtual ~GraphImpl() = default;

  std::shared_ptr<Graph::GraphData> &MutableGraphData() const { return graph_->graph_data_; }
  void SetGraph(const std::shared_ptr<Graph> &graph) { graph_ = graph; }
  void SetContext(const std::shared_ptr<Context> &context) { graph_context_ = context; }
  VectorRef GenerateInputsRef(const std::vector<tensor::TensorPtr> &inputs, const FuncGraphPtr &func_graph) {
    VectorRef results;
    std::size_t size = inputs.size();
    for (std::size_t i = 0; i < size; i++) {
      results.push_back(inputs[i]);
    }

    MS_EXCEPTION_IF_NULL(func_graph);
    std::vector<AnfNodePtr> graph_params = func_graph->parameters();
    std::size_t graph_params_size = graph_params.size();
    if (results.size() != graph_params_size) {
      // Maybe some default parameter
      for (std::size_t i = results.size(); i < graph_params_size; i++) {
        MS_EXCEPTION_IF_NULL(graph_params[i]);
        auto param_ptr = (graph_params[i])->cast_ptr<Parameter>();
        MS_EXCEPTION_IF_NULL(param_ptr);
        if (!param_ptr->has_default()) {
          MS_LOG(INTERNAL_EXCEPTION) << "Parameter[" << i << "] has no default param";
        }
        if (!param_ptr->default_param()->isa<tensor::Tensor>()) {
          MS_LOG(INTERNAL_EXCEPTION) << "Parameter[" << param_ptr->ToString()
                                     << "] is not initialized, need to call `.init_data()`";
        }
        results.push_back(param_ptr->default_param());
      }
    }
    return results;
  }

  uint32_t GetRootGraphIdFromActorInfo(const std::string &actor_info) {
    const std::string prefix = "kernel_graph_";
    auto pos = actor_info.find(prefix);
    if (pos == std::string::npos) {
      MS_LOG(INTERNAL_EXCEPTION) << "Cannot find prefix " << prefix << " from actor_info" << actor_info
                                 << ", failed to get graph id.";
    }
    std::string first_num = "";
    for (size_t i = prefix.size(); i < actor_info.size(); ++i) {
      if (actor_info[i] >= '0' && actor_info[i] <= '9') {
        first_num.push_back(actor_info[i]);
      } else {
        break;
      }
    }
    return std::stoul(first_num);
  }

  void GetModelInputsInfo(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                          std::vector<tensor::TensorPtr> *inputs, std::vector<std::string> *inputs_name) {
    MS_EXCEPTION_IF_NULL(kernel_graph);
    MS_EXCEPTION_IF_NULL(inputs);
    MS_EXCEPTION_IF_NULL(inputs_name);
    auto kernel_graph_inputs = kernel_graph->inputs();
    // find parameters of graph inputs
    for (size_t i = 0; i < kernel_graph_inputs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(kernel_graph_inputs[i]);
      if (!kernel_graph_inputs[i]->isa<Parameter>()) {
        MS_LOG(ERROR) << "Kernel graph inputs have anfnode which is not Parameter.";
        continue;
      }
      auto parameter = kernel_graph_inputs[i]->cast<ParameterPtr>();
      if (!common::AnfAlgo::IsParameterWeight(parameter)) {
        auto input_shape = AnfAlgo::GetOutputDeviceShape(parameter, 0);
        auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(parameter);
        MS_EXCEPTION_IF_NULL(kernel_build_info);
        auto data_type = kernel_build_info->GetOutputDeviceType(0);
        auto ms_tensor = std::make_shared<tensor::Tensor>(data_type, input_shape);
        inputs->push_back(ms_tensor);
        inputs_name->push_back(parameter->name());
      }
    }
  }

  void GetModelOutputsInfo(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                           std::vector<tensor::TensorPtr> *outputs, std::vector<std::string> *output_names) {
    MS_EXCEPTION_IF_NULL(kernel_graph);
    MS_EXCEPTION_IF_NULL(outputs);
    MS_EXCEPTION_IF_NULL(output_names);

    std::vector<tensor::TensorPtr> inputs;
    std::vector<std::string> input_names;
    GetModelInputsInfo(kernel_graph, &inputs, &input_names);

    VectorRef vector_outputs;
    std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node;
    session::KernelMapTensor node_to_tensor;
    auto anf_outputs = kernel_graph->outputs();
    for (auto &item : anf_outputs) {
      MS_EXCEPTION_IF_NULL(item);
      MS_LOG(INFO) << "Create node output[" << item->DebugString() << "]";
      vector_outputs.emplace_back(
        session::SessionBasic::CreateNodeOutputTensors(item, kernel_graph, inputs, &tensor_to_node, &node_to_tensor));
    }
    *outputs = TransformVectorRefToMultiTensor(vector_outputs);
    for (size_t i = 0; i < outputs->size(); i++) {
      output_names->push_back("output" + std::to_string(i));
    }
  }

  virtual Status Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) = 0;
  virtual Status Load(uint32_t device_id) = 0;

  virtual std::vector<MSTensor> GetInputs() = 0;
  virtual std::vector<MSTensor> GetOutputs() = 0;

  virtual bool CheckDeviceSupport(mindspore::DeviceType device_type) = 0;

 protected:
  std::shared_ptr<Graph> graph_;
  std::shared_ptr<Context> graph_context_;

  std::shared_ptr<compile::MindRTBackend> backend_;
  std::string actor_info_;
  std::weak_ptr<KernelGraph> kernel_graph_;
  std::weak_ptr<FuncGraph> func_graph_;
  uint32_t device_id_;
  std::vector<tensor::TensorPtr> inputs_info_;
  std::vector<tensor::TensorPtr> outputs_info_;
  std::vector<tensor::TensorPtr> last_inputs_;
  std::vector<tensor::TensorPtr> last_outputs_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  bool load_flag_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_GRAPH_GRAPH_IMPL_H
