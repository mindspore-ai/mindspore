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

#ifndef MINDSPORE_LITE_SRC_SCHEDULER_H_
#define MINDSPORE_LITE_SRC_SCHEDULER_H_

#include <utility>
#include <vector>
#include <map>
#include "src/sub_graph_kernel.h"
#include "src/inner_context.h"
#include "include/model.h"
#if SUPPORT_NPU
#include "src/runtime/agent/npu/optimizer/npu_pass_manager.h"
#endif

namespace mindspore::lite {
class Scheduler {
 public:
  Scheduler(const InnerContext *ctx, Model *src_model, std::vector<Tensor *> *src_tensors)
      : context_(ctx), src_model_(src_model), src_tensors_(src_tensors) {}
#if SUPPORT_NPU
  Scheduler(const InnerContext *ctx, Model *src_model, std::vector<Tensor *> *src_tensors,
            NPUManager *npu_manager = nullptr, NPUPassManager *npu_pass_manager = nullptr)
      : context_(ctx),
        src_model_(src_model),
        src_tensors_(src_tensors),
        npu_manager_(npu_manager),
        npu_pass_manager_(npu_pass_manager) {}
#endif
  ~Scheduler() = default;

  int Schedule(std::vector<kernel::LiteKernel *> *dst_kernels);

 private:
  void FindNodeInoutTensors(const lite::Model::Node &node, std::vector<Tensor *> *inputs,
                            std::vector<Tensor *> *outputs);
  // infer shape for a partial node
  int InferPartialShape(const lite::Model::Node *node, bool *infer_shape_interrupt);
  // infer shape for a node
  int InferNodeShape(const lite::Model::Node *node, bool *infer_shape_interrupt);
  // infer shape for a subgraph
  int InferSubGraphShape(size_t subgraph_index, bool *infer_shape_interrupt);

  // schedule a node to kernel according to context and kernels registered
  kernel::LiteKernel *FindBackendKernel(const std::vector<Tensor *> &in_tensors,
                                        const std::vector<Tensor *> &out_tensors, const Model::Node *node,
                                        TypeId prefer_data_type = kTypeUnknown);
  // schedule a partial node to a subgraph_kernel
  kernel::LiteKernel *SchedulePartialToKernel(const lite::Model::Node *src_node);
  // schedule a node to a kernel
  kernel::LiteKernel *ScheduleNodeToKernel(const lite::Model::Node *src_node, TypeId prefer_data_type = kTypeUnknown);
  // schedule a Model::SubGraph into a vector of kernel and subgraph_kernel
  int ScheduleSubGraphToKernels(size_t subgraph_index, std::vector<kernel::LiteKernel *> *dst_kernels,
                                std::vector<lite::Tensor *> *in_tensors, std::vector<lite::Tensor *> *out_tensors,
                                TypeId prefer_data_type = kTypeUnknown);

  // find in_kernels_ and out_kernels of kernel, sub_graph and nodes_ in sub_graph
  static void FindAllInoutKernels(const std::vector<kernel::LiteKernel *> &kernels);

  // vector<LiteKernel/SubGraphKernel> --> vector<SubGraphKernel>
  int ConstructSubGraphs(std::vector<kernel::LiteKernel *> *kernels);

  // create subgraph_kernel from a vector of kernel
  kernel::SubGraphKernel *CreateSubGraphKernel(const std::vector<kernel::LiteKernel *> &kernels,
                                               const std::vector<lite::Tensor *> *in_tensors,
                                               const std::vector<lite::Tensor *> *out_tensors,
                                               kernel::SubGraphType type);

  bool MergeOpIsReady(const kernel::LiteKernel *kernel, std::map<const kernel::LiteKernel *, bool> is_kernel_finish);

  bool KernelFitCurrentSubGraph(const kernel::SubGraphType subgraph_type, const kernel::LiteKernel &kernel);

  std::vector<kernel::LiteKernel *> FindAllSubGraphKernels(
    kernel::LiteKernel *head_kernel, std::map<const kernel::LiteKernel *, bool> *sinked_kernel_map);

  // other methods
  static TypeId GetFirstFp32Fp16OrInt8Type(const std::vector<Tensor *> &in_tensors);

  static void SetKernelTensorDataType(kernel::LiteKernel *kernel);

  static kernel::SubGraphType GetKernelSubGraphType(const kernel::LiteKernel *kernel);

  int RunPass(std::vector<kernel::LiteKernel *> *dst_kernels);

 protected:
  const InnerContext *context_ = nullptr;
  Model *src_model_ = nullptr;
  std::vector<Tensor *> *src_tensors_;
#if SUPPORT_NPU
  NPUManager *npu_manager_ = nullptr;
  NPUPassManager *npu_pass_manager_ = nullptr;
#endif
  std::vector<size_t> graph_output_node_indexes_;
  std::map<int, OpParameter *> op_parameters_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_SCHEDULER_H_
