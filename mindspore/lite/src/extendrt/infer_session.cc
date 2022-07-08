/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "extendrt/infer_session.h"

#include "extendrt/single_op_session.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/common_utils.h"
#include "backend/common/session/session_basic.h"
#include "backend/graph_compiler/graph_partition.h"
#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"

namespace mindspore {
static const std::vector<PrimitivePtr> ms_infer_cut_list = {prim::kPrimReturn,   prim::kPrimPartial,
                                                            prim::kPrimSwitch,   prim::kPrimMakeTuple,
                                                            prim::kPrimBpropCut, prim::kPrimSwitchLayer};
static bool is_infer_single_op = true;

class DefaultInferSession : public InferSession {
 public:
  DefaultInferSession() = default;
  virtual ~DefaultInferSession() = default;
  Status Init(const std::shared_ptr<Context> context) override;
  Status CompileGraph(FuncGraphPtr graph) override;
  Status RunGraph() override;
  Status RunGraph(const std::vector<tensor::TensorPtr> &inputs, std::vector<tensor::TensorPtr> *outputs) override;
  Status Resize(const std::vector<tensor::TensorPtr> &inputs, const std::vector<std::vector<int64_t>> &dims) override;

  std::vector<tensor::TensorPtr> GetOutputs() override;
  std::vector<tensor::TensorPtr> GetInputs() override;
  std::vector<std::string> GetOutputNames() override;
  std::vector<std::string> GetInputNames() override;
  tensor::TensorPtr GetOutputByTensorName(const std::string &tensorName) override;
  tensor::TensorPtr GetInputByTensorName(const std::string &name) override;

 private:
  session::SessionPtr session_basic_;
  KernelGraphPtr kernel_graph_;
  std::vector<KernelGraphPtr> kernel_graphs_;
};

Status DefaultInferSession::Init(const std::shared_ptr<Context> context) {
  MS_LOG(INFO) << "DefaultInferSession::Init";
  session_basic_ = std::make_shared<session::SessionBasic>();
  partition_ = std::make_shared<compile::GraphPartition>(ms_infer_cut_list, "ms");
  return kSuccess;
}
Status DefaultInferSession::CompileGraph(FuncGraphPtr graph) {
  MS_LOG(INFO) << "DefaultInferSession::CompileGraph";

  bool contain_multi_target = false;
  const auto &segments = partition_->Partition(graph, &contain_multi_target);

  for (auto &segment : segments) {
    FuncGraphPtr fg;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    auto kernel_seg_graph =
      session_basic_->ConstructKernelGraph(segment->nodes_, outputs, mindspore::device::DeviceType::kCPU);
    auto kernel_nodes = kernel_seg_graph->execution_order();

    MS_LOG(INFO) << "DefaultInferSession::CompileGraph Dump Kernels";
    for (const auto &kernel_node : kernel_nodes) {
      std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
      std::shared_ptr<kernel::CpuKernelMod> cpu_kernel_mod =
        kernel::Factory<kernel::CpuKernelMod>::Instance().Create(kernel_name);
      MS_LOG(INFO) << "DefaultInferSession::CompileGraph kernels " << kernel_name;
    }

    kernel_graphs_.push_back(kernel_seg_graph);
  }
  return kSuccess;
}

Status DefaultInferSession::RunGraph() { return kSuccess; }
Status DefaultInferSession::RunGraph(const std::vector<tensor::TensorPtr> &inputs,
                                     std::vector<tensor::TensorPtr> *outputs) {
  return kSuccess;
}
Status DefaultInferSession::Resize(const std::vector<tensor::TensorPtr> &inputs,
                                   const std::vector<std::vector<int64_t>> &dims) {
  return kSuccess;
}
std::vector<tensor::TensorPtr> DefaultInferSession::GetOutputs() { return std::vector<tensor::TensorPtr>(); }
std::vector<tensor::TensorPtr> DefaultInferSession::GetInputs() { return std::vector<tensor::TensorPtr>(); }
std::vector<std::string> DefaultInferSession::GetOutputNames() { return std::vector<std::string>(); }
std::vector<std::string> DefaultInferSession::GetInputNames() { return std::vector<std::string>(); }
tensor::TensorPtr DefaultInferSession::GetOutputByTensorName(const std::string &tensorName) { return nullptr; }
tensor::TensorPtr DefaultInferSession::GetInputByTensorName(const std::string &name) { return nullptr; }
std::shared_ptr<InferSession> InferSession::CreateSession(const std::shared_ptr<Context> context) {
  if (is_infer_single_op) {
    return std::make_shared<SingleOpInferSession>();
  }
  return std::make_shared<DefaultInferSession>();
}
}  // namespace mindspore
