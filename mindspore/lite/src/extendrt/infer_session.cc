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

#include "extendrt/session/single_op_session.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/common_utils.h"
#include "backend/graph_compiler/graph_partition.h"
#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"
#include "extendrt/utils/kernel_graph_utils.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "extendrt/delegate/factory.h"
#include "extendrt/session/factory.h"
#include "extendrt/delegate/plugin/tensorrt_executor_plugin.h"

namespace mindspore {
static const std::vector<PrimitivePtr> ms_infer_cut_list = {prim::kPrimReturn,   prim::kPrimPartial,
                                                            prim::kPrimSwitch,   prim::kPrimMakeTuple,
                                                            prim::kPrimBpropCut, prim::kPrimSwitchLayer};
static bool is_infer_single_op = true;
static bool is_use_lite_session = false;
// static bool is_use_tensorrt_delegate = true;

class DefaultInferSession : public InferSession {
 public:
  explicit DefaultInferSession(const std::shared_ptr<Context> &context) {}
  virtual ~DefaultInferSession() = default;
  Status Init(const std::shared_ptr<Context> &context) override;
  Status CompileGraph(FuncGraphPtr graph, const void *data = nullptr, size_t size = 0) override;
  Status RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) override;
  std::vector<MutableTensorImplPtr> GetOutputs() override;
  std::vector<MutableTensorImplPtr> GetInputs() override;
  std::vector<std::string> GetOutputNames() override;
  std::vector<std::string> GetInputNames() override;
  MutableTensorImplPtr GetOutputByTensorName(const std::string &tensorName) override;
  MutableTensorImplPtr GetInputByTensorName(const std::string &name) override;

 private:
  KernelGraphUtilsPtr kernel_graph_utils_;
  KernelGraphPtr kernel_graph_;
  std::vector<KernelGraphPtr> kernel_graphs_;
};

Status DefaultInferSession::Init(const std::shared_ptr<Context> &context) {
  MS_LOG(INFO) << "DefaultInferSession::Init";
  kernel_graph_utils_ = std::make_shared<mindspore::KernelGraphUtils>();
  partition_ = std::make_shared<compile::GraphPartition>(ms_infer_cut_list, "ms");
  return kSuccess;
}
Status DefaultInferSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size) {
  MS_LOG(INFO) << "DefaultInferSession::CompileGraph";
  return kSuccess;
}

Status DefaultInferSession::RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) {
  return kSuccess;
}
std::vector<MutableTensorImplPtr> DefaultInferSession::GetOutputs() { return {}; }
std::vector<MutableTensorImplPtr> DefaultInferSession::GetInputs() { return {}; }
std::vector<std::string> DefaultInferSession::GetOutputNames() { return std::vector<std::string>(); }
std::vector<std::string> DefaultInferSession::GetInputNames() { return std::vector<std::string>(); }
MutableTensorImplPtr DefaultInferSession::GetOutputByTensorName(const std::string &tensorName) { return nullptr; }
MutableTensorImplPtr DefaultInferSession::GetInputByTensorName(const std::string &name) { return nullptr; }
std::shared_ptr<InferSession> InferSession::CreateSession(const std::shared_ptr<Context> &context) {
  HandleContext(context);
  auto session_type = SelectSession(context);
  MS_LOG(DEBUG) << "Session type " << static_cast<int64_t>(session_type);
  return SessionRegistry::GetInstance().GetSession(session_type, context);
}

void InferSession::HandleContext(const std::shared_ptr<Context> &context) {
  if (!context) {
    return;
  }
  constexpr auto default_gpu_provider = "tensorrt";
  constexpr auto default_cpu_provider = "litert";
  auto device_infos = context->MutableDeviceInfo();
  for (auto &device_info : device_infos) {
    if (!device_info) {
      continue;
    }
    if (device_info->GetDeviceType() == kGPU) {
      auto gpu_device = device_info->Cast<GPUDeviceInfo>();
      if (!gpu_device) {
        continue;
      }
      auto provider = gpu_device->GetProvider();
      if (provider.empty() || provider == default_gpu_provider) {
        if (!lite::TensorRTExecutorPlugin::GetInstance().Register()) {
          MS_LOG_WARNING << "Failed to register TensorRT plugin";
          return;
        }
        gpu_device->SetProvider(default_gpu_provider);
      }
      continue;
    }

    if (device_info->GetDeviceType() == kCPU) {
      auto cpu_device = device_info->Cast<CPUDeviceInfo>();
      if (!cpu_device) {
        continue;
      }
      auto provider = cpu_device->GetProvider();
      if (provider.empty()) {
        cpu_device->SetProvider(default_cpu_provider);
      }
      continue;
    }
  }
}

SessionType InferSession::SelectSession(const std::shared_ptr<Context> &context) {
  if (context != nullptr) {
    auto &device_contexts = context->MutableDeviceInfo();
    for (auto device_context : device_contexts) {
      MS_EXCEPTION_IF_NULL(device_context);
      if (device_context->GetDeviceType() == kAscend) {
        return kSingleOpSession;
      }
      // if (device_context->GetDeviceType() == kGPU && is_use_tensorrt_delegate) {
      if (device_context->GetDeviceType() == kGPU || device_context->GetDeviceType() == kCPU) {
        return kDelegateSession;
      }
    }
  }

  if (is_infer_single_op) {
    return kSingleOpSession;
  }
  if (is_use_lite_session) {
    return kLiteInferSession;
  }
  return kDefaultSession;
}

static std::shared_ptr<InferSession> DefaultSessionCreator(const std::shared_ptr<Context> &ctx) {
  auto session = std::make_shared<DefaultInferSession>(ctx);
  session->Init(ctx);
  return session;
}
REG_SESSION(kDefaultSession, DefaultSessionCreator);
}  // namespace mindspore
