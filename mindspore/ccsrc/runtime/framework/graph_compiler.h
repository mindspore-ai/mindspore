/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_COMPILER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_COMPILER_H_

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <map>
#include <set>
#include "runtime/hardware/device_context.h"
#include "backend/session/session_basic.h"
#include "backend/session/session_factory.h"
#include "ir/tensor.h"

namespace mindspore {
using device::DeviceContext;
using session::CallBackFunc;
using session::GraphOutputInfo;
using session::InputTensorInfo;
using session::KernelGraph;
using session::KernelWithIndex;
using session::OpRunInfo;
using tensor::TensorPtr;

namespace runtime {
class GraphCompiler {
 public:
  GraphCompiler() { session_ = session::SessionFactory::Get().Create(kSessionBasic); }
  ~GraphCompiler() = default;

  // Construct kernel graph from anf nodes list and compile kernel graph in Graph mode,
  // the detailed implementation of compiling graph is in 'CompileGraphImpl'.
  GraphId CompileGraph(const AnfNodePtrList &nodes, const AnfNodePtrList &outputs, const DeviceContext *device_context);

  // Construct single op kernel graph and compile the kernel graph in PyNative mode.
  GraphId CompileGraph(const session::OpRunInfo &op_run_info, const GraphInfo &graph_info,
                       const std::vector<int64_t> *tensors_mask, std::vector<TensorPtr> *const input_tensors,
                       bool *single_op_cache_hit, const DeviceContext *device_context);

  // Get graph by graph id, if not exist return nullptr, used in Graph mode.
  KernelGraphPtr Fetch(GraphId graph_id) const;

  // Get graph by graph info, if not exist return nullptr, used in PyNative mode.
  KernelGraphPtr Fetch(const GraphInfo &graph_info) const;

  // The following four methods used in PyNative back propagation to split complete kernel graph to single
  // op graph, and these methods will be removed to class MindRTBackend after deleting session module.

  // Cache index for all parameter and output nodes of kernel graph, used to get parameter of single op and
  // recover output of original complete back propagation kernel graph.
  void GetParamAndOutputIndex(const KernelGraphPtr &graph, const std::vector<TensorPtr> &inputs,
                              VectorRef *const outputs, std::map<AnfNodePtr, size_t> *parameter_index,
                              std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes);

  // Get input tensors for single op compile and run, input tensors may convert from value node and parameter in graph
  // and prev kernel node's output.
  void GetSingleOpInputTensors(const CNodePtr &kernel, const std::map<KernelWithIndex, TensorPtr> &op_output,
                               const std::map<AnfNodePtr, size_t> &parameter_index,
                               const std::vector<TensorPtr> &graph_inputs, InputTensorInfo *const input_tensor_info);
  // Get one input tensor for single control op, such as bprop_cut.
  TensorPtr GetSingleOpInputTensorByIndex(const CNodePtr &kernel, const std::map<KernelWithIndex, TensorPtr> &op_output,
                                          const std::map<AnfNodePtr, size_t> &parameter_index,
                                          const std::vector<TensorPtr> &graph_inputs,
                                          InputTensorInfo *const input_tensor_info, size_t input_index);

  // Get OpRunInfo and GraphInfo for single op compile and run.
  void GetSingleOpRunInfoAndGraphInfo(const CNodePtr &kernel, const std::vector<TensorPtr> &input_tensors,
                                      OpRunInfo *const run_info, GraphInfo *const graph_info);

  // Calculate ref count of PyNative back propagation operators.
  void CalculateRefCount(const KernelGraphPtr &graph, std::map<KernelWithIndex, size_t> *ref_count) const;

  // Update ref count of PyNative back propagation operators.
  void UpdateRefCount(const std::set<KernelWithIndex> &input_kernels_with_index,
                      std::map<KernelWithIndex, size_t> *ref_count,
                      std::map<KernelWithIndex, tensor::TensorPtr> *op_output_map) const;

  // Handle single op output tensor and recover output of original complete kernel graph.
  void RecoverGraphOutput(const AnfNodePtr &kernel, const VectorRef &op_outputs,
                          const std::map<KernelWithIndex, size_t> &ref_count,
                          std::map<KernelWithIndex, TensorPtr> *op_output_map,
                          GraphOutputInfo *const graph_output_info) const;

  // Collect output tensors of back propagation graph for allreduce operators to average gradient,
  // used in PyNative distributed training mode.
  void AddGradAddrToBucket(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &grad_tensor);

  // Clear resource in bucket, such as useless tensors and device memory of all communication operators,
  // Bucket is used in PyNative distributed training mode, one bucket handles all resource to launch and sync allreduce
  // operator.
  void ClearAllBucket(const GraphId &graph_id);

  const std::vector<KernelWithIndex> &GetGraphOutputNodes(GraphId graph_id) const;

  // Register a summary callback function, which is called in the final stages of summary.
  void RegisterSummaryCallBackFunc(const CallBackFunc &callback) const;
  // Execute graph summary.
  void Summary(const std::vector<KernelGraphPtr> &graphs) const;

 private:
  DISABLE_COPY_AND_ASSIGN(GraphCompiler);

  // The implementation of compiling graph in Graph Mode, including optimizing graph,
  // setting operator info, creating kernel and transforming kernel graph to ActorSet.
  GraphId CompileGraphImpl(const KernelGraphPtr &graph, const DeviceContext *device_context) const;

  // Create device address for all anf nodes of graph.
  void CreateDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) const;

  // Single op kernel graph cache for PyNative mode.
  std::unordered_map<GraphInfo, KernelGraphPtr> run_op_graphs_;
  // Single op kernel graph output nodes cache for PyNative mode.
  std::unordered_map<GraphId, std::vector<KernelWithIndex>> run_op_graph_output_nodes_;

  // The member variable 'session_' will be removed after removing session module.
  // Now all the GraphCompiler share the same 'session_'.
  session::SessionPtr session_;
};

}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_COMPILER_H_
