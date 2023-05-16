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
#include <algorithm>
#include "src/common/log_util.h"
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/lite_kernel.h"
#include "src/common/tensor_util.h"
#include "src/extendrt/kernel/kernel_lib.h"
#include "src/extendrt/kernel/default_kernel_selector.h"
#include "src/litert/pass/format_pass/format_pass.h"
#include "tools/optimizer/graph/node_infershape.h"
#include "src/extendrt/graph_compiler/anfnode_tensor_adapter.h"

namespace mindspore {
namespace infer {
abstract::Kernel *SingleGraphScheduler::Schedule(const CompileResultPtr &node_list) {
  // infer shape
  auto infer_ret = FallBackInferShape(node_list);
  if (!infer_ret) {
    MS_LOG(ERROR) << "InferShape CompileResult node failed.";
    return nullptr;
  }

  execution_flow_ = std::make_shared<ExecutionFlow>();
  MSLITE_CHECK_PTR_RETURN(execution_flow_, nullptr);
  execution_flow_->SetInputs(node_list->GetInputs());
  execution_flow_->SetOutputs(node_list->GetOutputs());
  execution_flow_->SetTensors(node_list->GetTensors());
  auto schedule_ret = SelectKernel(node_list);
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
  auto kernel = execution_flow_->ConstructFusionKernel();
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Construct subgraph kernel failed.";
    return nullptr;
  }
  return kernel;
}

int SingleGraphScheduler::SelectKernel(const CompileResultPtr &node_list) {
  kernel::DefaultKernelSelector selector(&compile_option_);
  std::vector<abstract::Kernel *> kernels;
  for (const auto &node : node_list->GetNodes()) {
    MSLITE_CHECK_PTR_RETURN(node, lite::RET_NULL_PTR);
    auto lite_kernel = selector.CreateKernel({node->GetType(), node->GetKernelAttr(), compile_option_.format,
                                              node->GetBaseOperator(), node->GetCNode(), compile_option_.backend},
                                             node->GetInputs(), node->GetOutputs(), context_);
    if (lite_kernel == nullptr) {
      MS_LOG(ERROR) << "Create kernel for node: " << node->GetName() << " failed.";
      return lite::RET_NOT_SUPPORT;
    }
    auto *kernel_exec = new (std::nothrow) kernel::KernelExec(std::shared_ptr<kernel::LiteKernel>(lite_kernel));
    if (kernel_exec == nullptr) {
      MS_LOG(ERROR) << "Create kernel exec for node: " << node->GetName() << " failed.";
      return lite::RET_MEMORY_FAILED;
    }
    auto desc = kernel_exec->desc();
    desc.format = compile_option_.format;
    kernel_exec->set_desc(desc);
    kernel_exec->set_context(context_);
    kernels.push_back(kernel_exec);
  }
  execution_flow_->SetKernels(kernels);
  return lite::RET_OK;
}

bool SingleGraphScheduler::HandleWeightForKernels() {
  if (compile_option_.datatype != kNumberTypeFloat32 && compile_option_.datatype != kNumberTypeFloat16) {
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
      auto ret = CastConstTensorData(input, compile_option_.datatype, context_->device_and_pkg_support_fp16_);
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "Cast data for tensor: " << input->tensor_name() << " failed.";
        return false;
      }
    }
  }
  return true;
}

Status SingleGraphScheduler::OptimizeTranspose(std::vector<kernel::KernelExec *> *kernels) {
  auto tensors = execution_flow_->GetTensors();
  auto ret = lite::pass::RuntimeFormatPass(kernels, &tensors, compile_option_.format);
  if (ret != RET_OK) {
    MS_LOG(INFO) << "Run Optimize transpose pass failed.";
    return kLiteError;
  }
  return kSuccess;
}

namespace {
bool SetDTAndShapeFromAbTensorToLiteTensor(const AbstractBasePtr &abstract, lite::Tensor *tensor) {
  if (!utils::isa<mindspore::abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(ERROR) << "The abstract should be tensor, but got abstract : " << abstract;
    return false;
  }
  ShapeVector shape_vector;
  TypeId data_type = kTypeUnknown;
  auto ret = infer::TensorAdapter::GetDTAndShapeFromAbTensor(
    utils::cast<mindspore::abstract::AbstractTensorPtr>(abstract), &data_type, &shape_vector);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Get dtype and shape from abstract failed, abstract : " << abstract;
    return false;
  }
  std::vector<int32_t> int32_shape;
  std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(int32_shape),
                 [](const auto &shape) { return static_cast<int32_t>(shape); });
  tensor->set_data_type(data_type);
  tensor->set_shape(int32_shape);
  tensor->set_format(NHWC);
  return true;
}
}  // namespace

int SingleGraphScheduler::InferShapeByNNACL(CompileNode *node, OpParameter *op_parameter) {
  auto ret = KernelInferShape(node->GetInputs(), node->GetOutputs(), op_parameter, context_->allocator);
  if (ret != lite::RET_OK && ret != lite::RET_INFER_INVALID) {
    return ret;
  }
  if (op_parameters_.find(node->GetName()) == op_parameters_.end()) {
    op_parameters_[node->GetName()] = op_parameter;
  }
  return ret;
}

int SingleGraphScheduler::InferShapeByOps(CompileNode *node) {
  auto node_infer_shape = std::make_shared<opt::NodeInferShape>();
  if (node_infer_shape == nullptr) {
    MS_LOG(ERROR) << "create NodeInferShape manager failed.";
    return false;
  }
  auto cnode = node->GetCNode();
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr";
    return lite::RET_ERROR;
  }
  auto anf_prim = GetValueNode<std::shared_ptr<Primitive>>(cnode->input(0));
  if (anf_prim == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return lite::RET_ERROR;
  }
  (void)anf_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(static_cast<int64_t>(compile_option_.format)));
  auto ret = node_infer_shape->InferShapeByOps(cnode, true);
  if (ret != lite::RET_OK) {
    return ret;
  }
  // invalid is no need to sync output shape from abstract
  auto abstract = cnode->abstract();
  if (utils::isa<mindspore::abstract::AbstractSequencePtr>(abstract)) {
    auto elements = utils::cast<mindspore::abstract::AbstractSequencePtr>(abstract)->elements();
    if (elements.size() != node->OutputSize()) {
      MS_LOG(ERROR) << "The cnode output size: " << elements.size()
                    << " is not equal to lite tensors size: " << node->OutputSize();
      return lite::RET_ERROR;
    }
    for (size_t i = 0; i < elements.size(); i++) {
      if (!SetDTAndShapeFromAbTensorToLiteTensor(elements[i], node->GetOutput(i))) {
        MS_LOG(ERROR) << "Set tensor info from abstract failed, abstract : " << elements[i];
        return lite::RET_ERROR;
      }
    }
    return lite::RET_OK;
  }
  if (utils::isa<mindspore::abstract::AbstractTensorPtr>(abstract)) {
    if (!SetDTAndShapeFromAbTensorToLiteTensor(abstract, node->GetOutput(0))) {
      MS_LOG(ERROR) << "Set tensor info from abstract failed, abstract : " << abstract;
      return lite::RET_ERROR;
    }
    return lite::RET_OK;
  }
  MS_LOG(ERROR) << "Unsupported abstract type: " << abstract;
  return lite::RET_ERROR;
}

int SingleGraphScheduler::FallBackInferShape(const CompileResultPtr &node_list) {
  for (const auto &node : node_list->GetNodes()) {
    MSLITE_CHECK_PTR_RETURN(node, false);
    auto base_operator = node->GetBaseOperator();
    MSLITE_CHECK_PTR_RETURN(base_operator, false);
    if (compile_option_.format == Format::NHWC) {  // for efficient
      auto op_parameter = lite::OperatorPopulateRegistry::GetInstance()->CreatePopulateByOp(base_operator);
      if (op_parameter != nullptr) {
        auto ret = InferShapeByNNACL(node, op_parameter);
        if (ret != lite::RET_OK && ret != lite::RET_INFER_INVALID) {
          MS_LOG(ERROR) << "Infer kernel failed for op: " << node->GetName();
          return ret;
        }
      }
    } else {
      auto ret = InferShapeByOps(node);
      if (ret != lite::RET_OK && ret != lite::RET_INFER_INVALID) {
        MS_LOG(ERROR) << "Infer kernel failed for op: " << node->GetName();
        return ret;
      }
    }
  }
  return true;
}
}  // namespace infer
}  // namespace mindspore
