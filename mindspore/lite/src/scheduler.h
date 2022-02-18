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

#ifndef MINDSPORE_LITE_SRC_SCHEDULER_H_
#define MINDSPORE_LITE_SRC_SCHEDULER_H_

#include <utility>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <unordered_map>
#include <set>
#include <string>
#include "src/sub_graph_kernel.h"
#include "src/inner_context.h"
#include "include/model.h"
#include "src/scheduler_cb.h"
#ifndef DELEGATE_CLIP
#include "include/api/delegate.h"
#endif
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include "src/control_flow/control_flow_scheduler.h"
#endif

namespace mindspore::lite {
constexpr int kDefaultDeviceType = -1;
class Scheduler {
 public:
  Scheduler(InnerContext *ctx, const mindspore::Context *ms_ctx, Model *src_model, std::vector<Tensor *> *src_tensors,
            std::vector<Tensor *> *input_tensors, std::vector<Tensor *> *output_tensors, bool is_train_session,
            int *is_infershape, bool *is_control_flow, std::map<std::string, TypeId> *executions,
            std::shared_ptr<Delegate> delegate = nullptr, int delegate_device_type = -1)
      : context_(ctx),
        ms_context_(ms_ctx),
        src_model_(src_model),
        src_tensors_(src_tensors),
        inputs_(input_tensors),
        outputs_(output_tensors),
        is_train_session_(is_train_session),
        is_control_flow_(is_control_flow),
        is_infershape_(is_infershape),
        delegate_(delegate),
        delegate_device_type_(delegate_device_type),
        execution_plan_(executions) {}
  ~Scheduler() = default;
  int Schedule(std::vector<kernel::LiteKernel *> *dst_kernels);
  void SetupSchedulerCb(std::unique_ptr<SchedulerCb> cb) { sched_cb_ = std::move(cb); }
  void SetConfig(const std::map<std::string, std::map<std::string, std::string>> *config_info) {
    config_info_ = config_info;
  }
  std::vector<kernel::LiteKernel *> NonTailCallNodes();

 private:
  int SchedulePreProcess();
  int CheckInputParam(std::vector<kernel::LiteKernel *> *dst_kernels);
  void FindNodeInoutTensors(const Model::Node &node, std::vector<Tensor *> *inputs, std::vector<Tensor *> *outputs);
  Model::Node *NodeInputIsPartial(const Model::Node *node);
  int InferPartialShape(const Model::Node *node);
  int InferCallShape(const Model::Node *node);
  int InferNodeShape(const Model::Node *node);
  void FreeOpParameters();
  int InferSubGraphShape(size_t subgraph_index);
  // schedule a node to kernel according to context and kernels registered
  int HandleBuildinCpuKernelWeight(const kernel::SubGraphType belong_subgraph_type, const kernel::LiteKernel *kernel);
  kernel::LiteKernel *FindBackendKernel(const std::vector<Tensor *> &in_tensors,
                                        const std::vector<Tensor *> &out_tensors, const Model::Node *node,
                                        TypeId prefer_data_type = kTypeUnknown);
  int FindCpuKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                    OpParameter *op_parameter, const kernel::KernelKey &desc, TypeId kernel_data_type,
                    kernel::LiteKernel **kernel);
  int CheckCpuValid(const std::vector<kernel::LiteKernel *> *dst_kernels) const;
  void ResetByExecutionPlan(std::string node_name, TypeId *data_type);

#ifdef GPU_OPENCL
  int FindGpuKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                    OpParameter *op_parameter, const kernel::KernelKey &desc, kernel::LiteKernel **kernel,
                    TypeId prefer_data_type);
#endif
  int FindProviderKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                         const Model::Node *node, TypeId data_type, kernel::LiteKernel **kernel);

  int InitKernels(std::vector<kernel::LiteKernel *> dst_kernels);
  kernel::LiteKernel *SchedulePartialToKernel(const lite::Model::Node *src_node);
  // schedule a partial node to a subgraph_kernel
  std::vector<kernel::LiteKernel *> ScheduleSubGraphToSubGraphKernels(const int &subgraph_index);
  // schedule a node to a kernel
  kernel::LiteKernel *ScheduleNodeToKernel(const Model::Node *src_node, TypeId prefer_data_type = kTypeUnknown);
  // schedule a Model::Graph into a vector of subgraph_kernel
  int ScheduleGraphToKernels(std::vector<kernel::LiteKernel *> *dst_kernels, TypeId prefer_data_type = kTypeUnknown);
  // schedule a Model::SubGraph into a vector of kernel and subgraph_kernel
  int ScheduleSubGraphToKernels(size_t subgraph_index, std::vector<kernel::LiteKernel *> *dst_kernels,
                                std::vector<lite::Tensor *> *in_tensors, std::vector<lite::Tensor *> *out_tensors,
                                TypeId prefer_data_type = kTypeUnknown);
  // vector<LiteKernel/SubGraphKernel> --> vector<SubGraphKernel>
  int ConstructNormalSubGraphs(const std::vector<kernel::LiteKernel *> src_kernel,
                               std::vector<kernel::LiteKernel *> *dst_kernel,
                               std::map<const kernel::LiteKernel *, bool> *sinked_kernel_map);

  int ConstructSubGraphs(std::vector<kernel::LiteKernel *> *dst_kernel);

  // create subgraph_kernel from a vector of kernel
  std::vector<kernel::LiteKernel *> ScheduleMainSubGraphToKernels();
  kernel::LiteKernel *SchedulePartialToSubGraphKernel(const int &subgraph_index);
  kernel::SubGraphType PartialSubGraphType(const std::vector<kernel::LiteKernel *> &kernels);

  // other methods
  static TypeId GetFirstFp32Fp16OrInt8Type(const std::vector<Tensor *> &in_tensors);
  static void SetKernelTensorDataType(kernel::LiteKernel *kernel);
  int CopyPartialShapeToSubGraph(const lite::Model::Node *partial_node);
  int RestoreSubGraphInput(const lite::Model::Node *partial_node);

  bool IsControlFlowPattern(const lite::Model::Node &partial_node);
  STATUS DelQuantDTypeCastKernel(std::vector<kernel::LiteKernel *> *kernels);
#ifdef ENABLE_FP16
  int SubGraphPreferDataType(const int &subgraph_index, TypeId *prefer_data_type);
#endif
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  int InferSwitchShape(const Model::Node *node);
  Model::Node *NodeInputIsSwitchType(const Model::Node *node);
  bool SubGraphHasScheduled(const int &index);
  void SubGraphMarkScheduled(const int &index);
  int ConstructControlFlowMainGraph(std::vector<kernel::LiteKernel *> *kernels);
#endif

#ifndef DELEGATE_CLIP
  /* delegate related */
  int ReplaceDelegateKernels(std::vector<kernel::LiteKernel *> *dst_kernels);
  int InitDelegateKernels(std::vector<kernel::LiteKernel *> *dst_kernels);
#endif

#ifdef ENABLE_OPENGL_TEXTURE
  bool GetEnableGLTexture() { return context_->GetGpuInfo().enable_gl_texture_; }
  void *GetGLContext() { return context_->GetGpuInfo().gl_context_; }
  void *GetGLDisplay() { return context_->GetGpuInfo().gl_display_; }
#endif

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
  int *is_infershape_ = nullptr;
  std::unique_ptr<SchedulerCb> sched_cb_;
  std::map<kernel::Kernel *, const schema::Primitive *> primitives_;
  std::shared_ptr<Delegate> delegate_ = nullptr;
  int delegate_device_type_ = -1;
  std::deque<int> subgraphs_to_schedule_{};
  std::unordered_map<size_t, kernel::LiteKernel *> subgraph_index_subgraph_kernel_map_{};
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  std::set<int> scheduled_subgraph_index_{};
  std::unordered_map<kernel::LiteKernel *, size_t> partial_kernel_subgraph_index_map_{};
  std::set<lite::Model::Node *> partial_cnode_inferred_{};
  ControlFlowSchedulerPtr control_flow_scheduler_ = nullptr;
#endif
  int schema_version_ = SCHEMA_VERSION::SCHEMA_CUR;
  std::map<std::string, TypeId> *execution_plan_ = nullptr;
  const std::map<std::string, std::map<std::string, std::string>> *config_info_ = nullptr;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_SCHEDULER_H_
