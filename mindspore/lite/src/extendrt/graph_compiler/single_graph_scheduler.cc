/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/extendrt/graph_compiler/single_graph_scheduler.h"
#include "src/common/log_util.h"
#include "src/common/tensor_util.h"
#include "src/extendrt/graph_compiler/infershape_helper.h"
#include "src/extendrt/kernel/kernel_selector/kernel_selector.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/pass/format_pass/format_pass.h"
#include "src/litert/pass/format_pass/pass_utils.h"
#include "tools/optimizer/graph/node_infershape.h"
#include "src/common/draw/drawer.h"
#include "src/extendrt/kernel/nnacl/nnacl_base_kernel.h"
#include "src/extendrt/kernel/extendrt_kernel_exec.h"
#include "nnacl/format_transpose_parameter.h"
#include "extendrt/delegate/ascend_native/delegate.h"
#include "extendrt/delegate/factory.h"

namespace mindspore {
namespace lite {
InferKernel *SingleGraphScheduler::Schedule(const CompileResultPtr &node_list) {
  DrawDot(node_list.get(), "start_schedule");
  // infer shape
  MS_ASSERT(compile_option_ != nullptr);
  // try infer shape, if failed, will infer shape by kernel
  (void)GraphFallBackInferShape(node_list, compile_option_->graph_format, context_.get());
  DrawDot(node_list.get(), "fallback_infershape");

  execution_flow_ = std::make_shared<infer::ExecutionFlow>();
  MSLITE_CHECK_PTR_RETURN(execution_flow_, nullptr);
  execution_flow_->SetInputs(node_list->GetInputs());
  execution_flow_->SetOutputs(node_list->GetOutputs());
  execution_flow_->SetTensors(node_list->GetTensors());
  execution_flow_->SetContext(context_);
  auto schedule_ret = SelectKernel(node_list);
  if (schedule_ret != RET_OK) {
    MS_LOG(ERROR) << "Scheduler CompileResult to kernels failed.";
    return nullptr;
  }

  // append kernel with transpose
  auto kernel = execution_flow_->ConstructFusionKernel();
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Construct subgraph kernel failed.";
    return nullptr;
  }
  kernel->set_context(context_.get());
  DrawDot(kernel, "select_kernel");

  auto ret = OptimizeTranspose(kernel);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Optimize format of executionplan failed.";
    return nullptr;
  }

  auto infer_ret = kernel->InferShape();
  if (infer_ret != RET_OK && infer_ret != RET_INFER_INVALID) {
    MS_LOG(ERROR) << "InferShape SubGraph kernel failed.";
    return nullptr;
  }
  DrawDot(reinterpret_cast<kernel::SubGraphKernel *>(kernel), "kernel_infershape");
  return kernel;
}

void SingleGraphScheduler::CreateDelegateKernel(const std::shared_ptr<CompileNode> &node,
                                                mindspore::ExtendDelegate *delegate,
                                                std::vector<InferKernel *> *kernels) {
  auto delegate_kernel = delegate->CreateKernel({node->GetType(), node->GetKernelAttr(), compile_option_->graph_format,
                                                 compile_option_->backend, node->GetBaseOperator(), node->GetCNode()},
                                                node->GetInputs(), node->GetOutputs(), context_.get());
  if (delegate_kernel != nullptr) {
    auto kernel_exec = new kernel::ExtendRTKernelExec(delegate_kernel);
    kernel_exec->set_name(node->GetName());
    auto desc = kernel_exec->desc();
    desc.format = Format::NCHW;  // now all delegate should be nchw, but this will cause bad performance.
    kernel_exec->set_desc(desc);
    kernel_exec->set_context(this->context_.get());  // not safety
    kernels->push_back(kernel_exec);
  }
}

int SingleGraphScheduler::SelectKernel(const CompileResultPtr &node_list) {
  kernel_selector_ = kernel::CreateKernelSelector(compile_option_);
  const ConfigInfos config_infos;
  auto &device_contexts = ctx_->MutableDeviceInfo();
  if (device_contexts.empty()) {
    MS_LOG(ERROR) << "no context found";
    return RET_ERROR;
  }
  auto device_type = device_contexts.at(0)->GetDeviceType();
  auto provider = device_contexts.at(0)->GetProvider();
  auto delegate =
    DelegateRegistry<ExtendDelegate *>::GetInstance().GetDelegate(device_type, provider, ctx_, config_infos);
  std::vector<InferKernel *> kernels;
  for (const auto &node : node_list->GetNodes()) {
    MSLITE_CHECK_PTR_RETURN(node, RET_NULL_PTR);
    if ((delegate != nullptr) && (delegate->IsDelegateNode(node->GetCNode()))) {
      CreateDelegateKernel(node, delegate, &kernels);
      continue;
    }
    auto kernel_exec =
      kernel_selector_->CreateKernel({node->GetType(), node->GetKernelAttr(), compile_option_->graph_format,
                                      kernel::kBackendCPU, node->GetBaseOperator(), node->GetCNode()},
                                     node->GetInputs(), node->GetOutputs(), context_.get());
    if (kernel_exec == nullptr) {
      MS_LOG(ERROR) << "Create kernel exec for node: " << node->GetName() << " failed.";
      return RET_NOT_SUPPORT;
    }
    kernel_exec->set_name(node->GetName());
    kernels.emplace_back(kernel_exec);
  }
  execution_flow_->SetKernels(kernels);
  return RET_OK;
}

bool SingleGraphScheduler::HandleWeightForKernels() {
  if (compile_option_->datatype != kNumberTypeFloat32 && compile_option_->datatype != kNumberTypeFloat16) {
    return true;
  }
  auto kernels = execution_flow_->GetKernels();
  for (const auto &kernel : kernels) {
    for (const auto &input : kernel->in_tensors()) {
      // only cast const tensor
      if (!input->IsConst()) {
        continue;
      }
      // only support fp32->fp16 or fp16->fp32
      if (input->data_type() != kNumberTypeFloat32 && input->data_type() != kNumberTypeFloat16) {
        continue;
      }
      auto ret = CastConstTensorData(input, compile_option_->datatype, context_->device_and_pkg_support_fp16_);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Cast data for tensor: " << input->tensor_name() << " failed.";
        return false;
      }
    }
  }
  return true;
}

namespace {
kernel::KernelExec *CreateFormatTransFunc(InferTensor *input, InferTensor *output,
                                          const pass::TransInfoPair &trans_info, const std::string &name,
                                          const InferContext *ctx, const kernel::KernelKey &desc) {
  auto param = reinterpret_cast<FormatTransposeParameter *>(malloc(sizeof(FormatTransposeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Malloc FormatTransposeParameter failed.";
    return nullptr;
  }
  (void)memset(param, 0, sizeof(FormatTransposeParameter));
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_FormatTranspose);
  param->src_format_ = static_cast<FormatC>((trans_info.src_format_));
  param->dst_format_ = static_cast<FormatC>((trans_info.dst_format_));
  kernel::KernelKey format_transpose_key = desc;
  format_transpose_key.type = schema::PrimitiveType_FormatTranspose;
  format_transpose_key.format = NHWC;
  format_transpose_key.data_type = input->data_type();

  auto lite_kernel = KernelRegistry::GetInstance()->GetLiteKernel({input}, {output}, ctx, &format_transpose_key,
                                                                  reinterpret_cast<OpParameter *>(param));
  if (lite_kernel == nullptr) {
    MS_LOG(ERROR) << "Create format-transpose lite-kernel failed.";
    free(param);
    return nullptr;
  }
  auto base_kernel = new (std::nothrow) kernel::NNACLBaseKernel(std::shared_ptr<kernel::LiteKernel>(lite_kernel));
  if (base_kernel == nullptr) {
    MS_LOG(ERROR) << "Create format-transpose base-kernel failed.";
    return nullptr;
  }
  auto *kernel_exec = new (std::nothrow) kernel::ExtendRTKernelExec(std::shared_ptr<kernel::MSKernel>(base_kernel));
  if (kernel_exec == nullptr) {
    MS_LOG(ERROR) << "Create format-transpose kernel-exec failed.";
    return nullptr;
  }
  kernel_exec->set_desc(format_transpose_key);
  kernel_exec->set_context(ctx);
  kernel_exec->set_name(name);
  return kernel_exec;
}
}  // namespace

Status SingleGraphScheduler::OptimizeTranspose(kernel::SubGraphKernel *kernel) {
  std::vector<kernel::KernelExec *> subgraph_list = {kernel};
  auto ret =
    pass::DoFormatPass(&subgraph_list, &kernel->tensors(), compile_option_->graph_format, CreateFormatTransFunc);
  if (ret != RET_OK) {
    MS_LOG(INFO) << "Run Optimize transpose pass failed.";
    return kLiteError;
  }
  return kSuccess;
}
}  // namespace lite
}  // namespace mindspore
