/**
 * Copyright 2023  uawei Technologies Co., Ltd
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

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "src/extendrt/session/memory_offload_session.h"
#include "plugin/factory/ms_factory.h"
#include "src/extendrt/session/factory.h"
#include "src/extendrt/memory_offload/infer_strategy_builder.h"
#include "src/extendrt/utils/func_graph_utils.h"
#include "mindspore/lite/src/common/common.h"

namespace mindspore::lite {
Status MemoryOffloadInferSession::Init(const std::shared_ptr<Context> &context, const ConfigInfos &config_info) {
  context_ = context;
  return SingleOpInferSession::Init(context, config_info);
}

kernel::KernelModKernel *MemoryOffloadInferSession::BuildCustomAscendKernelImpl(
  const CNodePtr &cnode, const lite::CompileNodePtr &compile_node) {
  auto kernel_name = lite::kNameCustomAscend;
  std::shared_ptr<kernel::KernelMod> kernel_mod = kernel::Factory<kernel::KernelMod>::Instance().Create(kernel_name);
  if (kernel_mod == nullptr) {
    MS_LOG(ERROR) << "Kernel mod is nullptr, kernel name: " << kernel_name;
    return nullptr;
  }

  kernel_mod->SetDevicedId(device_id_);
  mindspore::kernel::BaseOperatorPtr base_operator;
  if (!FuncGraphUtils::GetCNodeOperator(cnode, &base_operator)) {
    MS_LOG(ERROR) << "Failed to create operator for cnode " << cnode->fullname_with_scope();
    return nullptr;
  }
  SetCustomAscendOpAttrs(base_operator);

  auto lite_kernel_mod =
    new (std::nothrow) kernel::KernelModKernel(kernel_mod, base_operator, compile_node->GetCNode(),
                                               compile_node->GetInputs(), compile_node->GetOutputs(), nullptr);
  if (lite_kernel_mod == nullptr) {
    MS_LOG(ERROR) << "new kernel failed " << kernel_name;
    return nullptr;
  }

  auto ret = lite_kernel_mod->Prepare();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "kernel prepare failed " << kernel_name;
    delete lite_kernel_mod;
    return nullptr;
  }

  MS_LOG(INFO) << "create Kernel: " << kernel_name << " succ";
  return lite_kernel_mod;
}

Status MemoryOffloadInferSession::BuildCustomAscendKernel(const CNodePtr &cnode, CompileNodePtr compile_node) {
  auto kernel = BuildCustomAscendKernelImpl(cnode, std::move(compile_node));
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Build ascend kernel failed for node: " << cnode->fullname_with_scope();
    return kLiteError;
  }
  kernels_.push_back(kernel);

  return kSuccess;
}

Status MemoryOffloadInferSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size, uint32_t *) {
  MS_LOG(INFO) << "MemoryOffloadInferSession::CompileGraph";
  auto compile_option = std::make_shared<CompileOption>();
  compile_option->graph_format = NCHW;
  lite::CompileResultBuilder compiler(compile_option);
  lite::CompileResultPtr compile_result_ = compiler.Build(graph);
  if (compile_result_ == nullptr) {
    MS_LOG(ERROR) << "Failed to build compile result";
    return kLiteError;
  }
  for (const auto &node : compile_result_->GetNodes()) {
    auto ret = BuildCustomAscendKernel(node->GetCNode(), node);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Failed to Build custom ascend kernel";
      return ret;
    }
  }
  MemoryOffloadInferStrategyBuilder strategy_builder;
  auto strategy_ = strategy_builder.Build(compile_result_, swap_context_);
  if (strategy_ == nullptr) {
    MS_LOG(ERROR) << "Failed to build strategy";
    return kLiteError;
  }

  return kSuccess;
}

static std::shared_ptr<InferSession> MemoryOffloadSessionCreator(const std::shared_ptr<Context> &ctx,
                                                                 const ConfigInfos &config_infos) {
  auto session = std::make_shared<MemoryOffloadInferSession>();
  MS_EXCEPTION_IF_NULL(session);
  session->Init(ctx);
  session->SetConfigInfo(config_infos);
  return session;
}

REG_SESSION(kMemoryOffloadSession, MemoryOffloadSessionCreator);
}  // namespace mindspore::lite
