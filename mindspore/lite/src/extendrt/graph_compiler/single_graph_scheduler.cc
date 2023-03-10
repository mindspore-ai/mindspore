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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/lite_kernel.h"
#include "src/litert/kernel_exec_util.h"
#include "src/common/tensor_util.h"
#include "src/extendrt/lite_kernel_mod.h"
#include "src/extendrt/graph_compiler/cnode_infer_manager.h"

namespace mindspore {
namespace infer {
ExecutionFlowPtr SingleGraphScheduler::Schedule(const CompileResultPtr &node_list) {
  // infer shape
  auto infer_ret = InferShape(node_list);
  if (!infer_ret) {
    MS_LOG(ERROR) << "InferShape CompileResult node failed.";
    return nullptr;
  }

  execution_plan_ = std::make_shared<ExecutionFlow>();
  MSLITE_CHECK_PTR_RETURN(execution_plan_, nullptr);
  execution_plan_->SetInputs(node_list->GetInputs());
  execution_plan_->SetOutputs(node_list->GetOutputs());
  graph_arch_ = kernel::kCPU;
  graph_data_type_ = kNumberTypeFloat32;
  // select kernel
  auto schedule_ret = ScheduleToKernels(node_list);
  if (schedule_ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Scheduler CompileResult to kernels failed.";
    return nullptr;
  }

  // fp16/fp32 weight, transpose weight
  auto cast_ret = HandleWeightForKernels();
  if (!cast_ret) {
    MS_LOG(ERROR) << "Handle weight for kernels failed.";
    return nullptr;
  }

  // append kernel with transpose
  // optimize transpose
  std::cout << execution_plan_->Dump() << std::endl;
  return execution_plan_;
}

int SingleGraphScheduler::ScheduleToKernels(const CompileResultPtr &node_list) {
  // todo, init graph_arch_, graph_data_type_, the SingleGraph has certain arch and data_type.
  std::vector<abstract::Kernel *> kernels;
  for (const auto &node : node_list->GetNodes()) {
    MSLITE_CHECK_PTR_RETURN(node, lite::RET_NULL_PTR);
    auto kernel = CreateKernel(node);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "Create kernel for node: " << node->GetName();
      return lite::RET_NULL_PTR;
    }
    kernels.push_back(kernel);
  }
  execution_plan_->SetKernels(kernels);
  return lite::RET_OK;
}

abstract::Kernel *SingleGraphScheduler::CreateKernel(const CompileNode *compile_node) {
  auto base_operator = compile_node->GetBaseOperator();
  kernel::KernelExec *kernel_exec = nullptr;
  kernel::KernelKey desc{graph_arch_, graph_data_type_, DEFAULT_FORMAT, PrimType_NONE};
  if (op_parameters_.find(compile_node->GetName()) != op_parameters_.end()) {
    // select lite kernel
    auto op_parameter = op_parameters_[compile_node->GetName()];
    MSLITE_CHECK_PTR_RETURN(op_parameter, nullptr);
    desc.type = op_parameter->type_;
    desc.format = NHWC;
    auto ret = lite::KernelRegistry::GetInstance()->GetKernelExec(compile_node->GetInputs(), compile_node->GetOutputs(),
                                                                  context_, nullptr, desc, op_parameter, &kernel_exec);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Get kernel from LiteRT failed for op: " << compile_node->GetName();
      return nullptr;
    }
  } else {
    // select core/ops kernel
    kernel_exec = kernel::FindKernelMod(compile_node->GetCNode(), base_operator, compile_node->GetInputs(),
                                        compile_node->GetOutputs(), context_);
    if (kernel_exec == nullptr) {
      MS_LOG(ERROR) << "Get kernel from KernelMod failed for op: " << compile_node->GetName();
      return nullptr;
    }
    desc.format = NCHW;  // set format NCHW to insert transpose kernel
    kernel_exec->set_desc(desc);
  }
  kernel_exec->set_name(compile_node->GetName());
  auto ret = kernel::KernelExecUtil::SetKernelTensorDataType(kernel_exec);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set tensor data type for kernel " << kernel_exec->name();
    delete kernel_exec;
    return nullptr;
  }
  return kernel_exec;
}

bool SingleGraphScheduler::HandleWeightForKernels() {
  if (graph_data_type_ != kNumberTypeFloat32 && graph_data_type_ != kNumberTypeFloat16) {
    return true;
  }
  auto kernels = execution_plan_->GetKernels();
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
      auto ret = CastConstTensorData(input, graph_data_type_, context_->device_and_pkg_support_fp16_);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "Cast data for tensor: " << input->tensor_name() << " failed.";
        return false;
      }
    }
  }
  return true;
}

bool SingleGraphScheduler::AppendKernelToPlan(kernel::KernelExec *kernel) { return false; }

bool SingleGraphScheduler::OptimizeTranspose(const std::vector<kernel::KernelExec *> &kernels) { return false; }

bool SingleGraphScheduler::InferShape(const CompileResultPtr &node_list) {
  for (const auto &node : node_list->GetNodes()) {
    MSLITE_CHECK_PTR_RETURN(node, false);
    auto base_operator = node->GetBaseOperator();
    MSLITE_CHECK_PTR_RETURN(base_operator, false);
    auto op_parameter = lite::OperatorPopulateRegistry::GetInstance()->CreatePopulateByOp(base_operator);
    if (op_parameter != nullptr) {
      auto ret = KernelInferShape(node->GetInputs(), node->GetOutputs(), op_parameter, context_->allocator);
      if (ret != lite::RET_OK && ret != lite::RET_INFER_INVALID) {
        MS_LOG(ERROR) << "Infer OpParameter kernel failed for op: " << node->GetName();
        return false;
      }
      if (op_parameters_.find(node->GetName()) == op_parameters_.end()) {
        op_parameters_[node->GetName()] = op_parameter;
      }
    } else {
      auto ret = CNodeInferShape(node->GetCNode(), node->GetOutputs());
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "Infer CNode kernel failed for op: " << node->GetName();
        return false;
      }
    }
  }
  return true;
}
}  // namespace infer
}  // namespace mindspore
