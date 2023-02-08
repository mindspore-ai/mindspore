/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_SCHEDULER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_SCHEDULER_H_

#include <utility>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <unordered_map>
#include <set>
#include <string>
#include "src/litert/sub_graph_kernel.h"
#include "src/litert/inner_context.h"
#include "include/model.h"
#include "src/litert/scheduler_cb.h"
#include "include/api/delegate.h"
#include "src/control_flow/control_flow_scheduler.h"
#include "src/litert/runtime_shape_fusion_pass.h"

namespace mindspore::lite {
constexpr int kDefaultDeviceType = -1;
class Scheduler {
 public:
  Scheduler(InnerContext *ctx, const mindspore::Context *ms_ctx, Model *src_model, std::vector<Tensor *> *src_tensors,
            std::vector<Tensor *> *input_tensors, std::vector<Tensor *> *output_tensors, bool is_train_session,
            int *is_infershape, bool *is_control_flow, bool *infer_along_running,
            std::map<std::string, TypeId> *executions, std::shared_ptr<Delegate> delegate = nullptr,
            int delegate_device_type = -1)
      : context_(ctx),
        ms_context_(ms_ctx),
        src_model_(src_model),
        src_tensors_(src_tensors),
        inputs_(input_tensors),
        outputs_(output_tensors),
        is_train_session_(is_train_session),
        is_control_flow_(is_control_flow),
        infer_along_running_(infer_along_running),
        is_infershape_(is_infershape),
        delegate_(delegate),
        delegate_device_type_(delegate_device_type),
        execution_plan_(executions) {}
  ~Scheduler() = default;
  int Schedule(std::vector<kernel::KernelExec *> *dst_kernels);
  void SetupSchedulerCb(std::unique_ptr<SchedulerCb> cb) { sched_cb_ = std::move(cb); }
  void SetConfig(const std::map<std::string, std::map<std::string, std::string>> *config_info) {
    config_info_ = config_info;
  }
  std::vector<kernel::KernelExec *> NonTailCallNodes();

 private:
  bool CheckRunNCXPass();
  int SchedulePreProcess();
  int CheckInputParam(const std::vector<kernel::KernelExec *> *dst_kernels) const;
  void FindNodeInoutTensors(const LiteGraph::Node &node, std::vector<Tensor *> *inputs, std::vector<Tensor *> *outputs);
  LiteGraph::Node *NodeInputIsPartial(const LiteGraph::Node *node);
  int InferPartialShape(const LiteGraph::Node *node);
  int InferCallShape(const LiteGraph::Node *node);
  int InferNodeShape(const LiteGraph::Node *node);
  void FreeOpParameters();
  int InferSubGraphShape(size_t subgraph_index);
  // schedule a node to kernel according to context and kernels registered
  int HandleBuildinCpuKernelWeight(const kernel::SubGraphType &belong_subgraph_type, const kernel::KernelExec *kernel);
  kernel::KernelExec *FindBackendKernel(const std::vector<Tensor *> &in_tensors,
                                        const std::vector<Tensor *> &out_tensors, const LiteGraph::Node *node,
                                        TypeId prefer_data_type = kTypeUnknown);
  int FindCpuKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                    OpParameter *op_parameter, const kernel::KernelKey &desc, TypeId kernel_data_type,
                    kernel::KernelExec **kernel);
  int CheckCpuValid(const std::vector<kernel::KernelExec *> *dst_kernels) const;
  void ResetByExecutionPlan(std::string node_name, TypeId *data_type);

#ifdef GPU_OPENCL
  int FindGpuKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                    OpParameter *op_parameter, const kernel::KernelKey &desc, kernel::KernelExec **kernel,
                    TypeId prefer_data_type);
#endif
  int FindProviderKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                         const LiteGraph::Node *node, TypeId data_type, kernel::KernelExec **kernel);

  int InitKernels(std::vector<kernel::KernelExec *> &&dst_kernels);
  kernel::KernelExec *SchedulePartialToKernel(const lite::LiteGraph::Node *src_node);
  // schedule a partial node to a subgraph_kernel
  std::vector<kernel::KernelExec *> ScheduleSubGraphToSubGraphKernels(const int &subgraph_index);
  // schedule a node to a kernel
  kernel::KernelExec *ScheduleNodeToKernel(const LiteGraph::Node *src_node, TypeId prefer_data_type = kTypeUnknown);
  // schedule a Model::Graph into a vector of subgraph_kernel
  int ScheduleGraphToKernels(std::vector<kernel::KernelExec *> *dst_kernels, TypeId prefer_data_type = kTypeUnknown);
  // schedule a LiteGraph::SubGraph into a vector of kernel and subgraph_kernel
  int ScheduleSubGraphToKernels(size_t subgraph_index, std::vector<kernel::KernelExec *> *dst_kernels,
                                std::vector<lite::Tensor *> *in_tensors, std::vector<lite::Tensor *> *out_tensors,
                                TypeId prefer_data_type = kTypeUnknown);
  // vector<KernelExec/SubGraphKernel> --> vector<SubGraphKernel>
  int ConstructNormalSubGraphs(const std::vector<kernel::KernelExec *> &src_kernel,
                               std::vector<kernel::KernelExec *> *dst_kernel,
                               std::map<const kernel::KernelExec *, bool> *sinked_kernel_map);

  int ConstructSubGraphs(std::vector<kernel::KernelExec *> *dst_kernel);

  // create subgraph_kernel from a vector of kernel
  std::vector<kernel::KernelExec *> ScheduleMainSubGraphToKernels();
  kernel::KernelExec *SchedulePartialToSubGraphKernel(const int &subgraph_index);
  kernel::SubGraphType PartialSubGraphType(const std::vector<kernel::KernelExec *> &kernels);

  // other methods
  static TypeId GetFirstFp32Fp16OrInt8Type(const std::vector<Tensor *> &in_tensors);
  int CopyPartialShapeToSubGraph(const lite::LiteGraph::Node *partial_node);
  int RestoreSubGraphInput(const lite::LiteGraph::Node *partial_node);

  bool IsControlFlowPattern(const lite::LiteGraph::Node &partial_node);
  STATUS DelQuantDTypeCastKernel(std::vector<kernel::KernelExec *> *kernels);
#ifdef ENABLE_FP16
  int SubGraphPreferDataType(const int &subgraph_index, TypeId *prefer_data_type);
#endif
  int InferSwitchShape(const LiteGraph::Node *node);
  LiteGraph::Node *NodeInputIsSwitchType(const LiteGraph::Node *node);
  bool SubGraphHasScheduled(const int &index);
  void SubGraphMarkScheduled(const int &index);
  int ConstructControlFlowMainGraph(std::vector<kernel::KernelExec *> *kernels);

#ifndef DELEGATE_CLIP
  /* delegate related */
  int ReplaceDelegateKernels(std::vector<kernel::KernelExec *> *dst_kernels);
  int InitDelegateKernels(std::vector<kernel::KernelExec *> *dst_kernels);
#else
  int InitDelegateKernels(std::vector<kernel::KernelExec *> *dst_kernels) { return RET_OK; }
#endif

  bool GetEnableGLTexture() const { return context_->GetDeviceInfo(DT_GPU).gpu_device_info_.enable_gl_texture_; }
  void *GetGLContext() const { return context_->GetDeviceInfo(DT_GPU).gpu_device_info_.gl_context_; }
  void *GetGLDisplay() const { return context_->GetDeviceInfo(DT_GPU).gpu_device_info_.gl_display_; }

 protected:
  InnerContext *context_ = nullptr;
  const mindspore::Context *ms_context_ = nullptr;
  Model *src_model_ = nullptr;
  std::vector<Tensor *> *src_tensors_;
  std::vector<Tensor *> *inputs_;
  std::vector<Tensor *> *outputs_;
  std::vector<mindspore::MSTensor> ms_inputs_;
  std::vector<mindspore::MSTensor> ms_outputs_;
  std::vector<size_t> graph_output_node_indexes_;
  std::map<int, OpParameter *> op_parameters_;
  bool is_train_session_ = false;
  bool *is_control_flow_ = nullptr;
  bool *infer_along_running_ = nullptr;
  int *is_infershape_ = nullptr;
  std::unique_ptr<SchedulerCb> sched_cb_;
  std::map<kernel::Kernel *, const schema::Primitive *> primitives_;
  std::shared_ptr<Delegate> delegate_ = nullptr;
  int delegate_device_type_ = -1;
  std::deque<int> subgraphs_to_schedule_{};
  std::unordered_map<size_t, kernel::KernelExec *> subgraph_index_subgraph_kernel_map_{};
  std::set<int> scheduled_subgraph_index_{};
  std::unordered_map<kernel::KernelExec *, size_t> partial_kernel_subgraph_index_map_{};
  std::set<lite::LiteGraph::Node *> partial_cnode_inferred_{};
  ControlFlowSchedulerPtr control_flow_scheduler_ = nullptr;
  int schema_version_ = SCHEMA_VERSION::SCHEMA_CUR;
  std::map<std::string, TypeId> *execution_plan_ = nullptr;
  const std::map<std::string, std::map<std::string, std::string>> *config_info_ = nullptr;
  std::shared_ptr<ShapeFusionPass> shape_fusion_pass_ = nullptr;
  std::vector<int> infer_subgraph_index_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_RUNTIME_SCHEDULER_H_
