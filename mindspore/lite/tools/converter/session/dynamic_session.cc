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

#include "tools/converter/session/dynamic_session.h"
#include "src/litert/scheduler.h"
#include "src/litert/kernel_exec_util.h"

namespace mindspore {
namespace lite {
int DynamicSession::CompileGraph(Model *model) {
  if (model->graph_.sub_graphs_.size() != 1) {
    MS_LOG(ERROR) << "Currently, don't support control-flow model.";
    return RET_NOT_SUPPORT;
  }
  auto ret = Init(context_);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "Init Context failed.");
  MS_CHECK_TRUE_MSG(context_->allocator != nullptr, RET_NULL_PTR, "Allocator is a nullptr.");
  ret = ConvertTensors(model);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "ConvertTensors failed.");
  InitGraphInputTensors(model);
  InitGraphOutputTensors(model);
  // scheduler kernels
  Scheduler scheduler(context_.get(), ms_context_, model, &tensors_, &inputs_, &outputs_, is_train_session_,
                      &is_infershape_, &is_control_flow_, &infer_along_running_, execution_plan_, delegate_,
                      delegate_device_type_);
  scheduler.SetupSchedulerCb(std::move(sched_cb_));
  scheduler.SetConfig(config_info_);
  ret = scheduler.Schedule(&kernels_);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "Schedule failed.");
  InitGraphInOutTensorsMap(model);
  for (auto tensor : tensors_) {
    tensor->set_allocator(context_->allocator);
  }
  auto kernels = kernels_;
  kernels_.clear();
  for (auto kernel : kernels) {
    if (kernel->subgraph_type() != kernel::kNotSubGraph) {
      auto nodes = reinterpret_cast<kernel::SubGraphKernel *>(kernel)->nodes();
      kernels_.insert(kernels_.end(), nodes.begin(), nodes.end());
    } else {
      kernels.push_back(kernel);
    }
  }
  // find in_kernels and out_kernels between subgraph kernels
  kernel::KernelExecUtil::FindAllInoutKernels(kernels_);
  for (auto kernel : kernels_) {
    kernel->InitOutTensorInitRefCount();
    ret = kernel->Prepare();
    MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "Prepare failed.");
  }
  return RET_OK;
}

std::vector<Tensor *> DynamicSession::GetInTensors(int index) {
  if (index < 0 || static_cast<size_t>(index) >= kernels_.size()) {
    return {};
  }
  return kernels_[index]->in_tensors();
}

std::vector<Tensor *> DynamicSession::GetOutTensors(int index) {
  if (index < 0 || static_cast<size_t>(index) >= kernels_.size()) {
    return {};
  }
  return kernels_[index]->out_tensors();
}

bool DynamicSession::CheckTensorIsVar(Tensor *tensor) {
  auto category = tensor->category();
  return category != CONST_TENSOR && category != CONST_SCALAR &&
         weights_name_.find(tensor->tensor_name()) == weights_name_.end();
}

int DynamicSession::PreProcess(kernel::KernelExec *kernel, std::vector<void *> *in_datas,
                               std::vector<void *> *out_datas) {
  for (auto tensor : kernel->in_tensors()) {
    if (CheckTensorIsVar(tensor)) {
      in_datas->push_back(tensor->data());
    }
  }
  auto ret =
    lite::KernelInferShape(kernel->in_tensors(), kernel->out_tensors(), kernel->op_parameter(), context_->allocator);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "InferShape failed.");
  ret = kernel->ReSize();
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "Resize failed.");
  for (auto out_tensor : kernel->out_tensors()) {
    out_tensor->FreeData();
    auto shape = out_tensor->shape();
    if (std::any_of(shape.begin(), shape.end(), [](int dim) { return dim < 0; })) {
      MS_LOG(ERROR) << "Don't support undetermined shape.";
      return RET_NOT_SUPPORT;
    }
    auto unit_data_size = out_tensor->Size();
    auto data = context_->allocator->Malloc(unit_data_size * sample_num_);
    MS_CHECK_TRUE_MSG(data != nullptr, RET_NULL_PTR, "Malloc data for out-tensor failed.");
    out_tensor->set_data(data);
    out_tensor->ResetRefCount();
    out_datas->push_back(data);
  }
  return RET_OK;
}

void DynamicSession::PrepareInOuts(kernel::KernelExec *kernel, const std::vector<void *> &ins,
                                   const std::vector<void *> &outs, int sample_index) {
  int i = 0;
  for (auto tensor : kernel->in_tensors()) {
    tensor->IncRefCount();
    if (CheckTensorIsVar(tensor)) {
      tensor->set_allocator(context_->allocator);
      tensor->set_data(static_cast<uint8_t *>(ins[i++]) + tensor->Size() * sample_index, false);
      tensor->set_allocator(nullptr);
    }
  }
  int j = 0;
  for (auto tensor : kernel->out_tensors()) {
    tensor->set_data(static_cast<uint8_t *>(outs[j++]) + tensor->Size() * sample_index, false);
  }
}

void DynamicSession::ResetInOuts(kernel::KernelExec *kernel, const std::vector<void *> &ins,
                                 const std::vector<void *> &outs) {
  int i = 0;
  for (auto tensor : kernel->in_tensors()) {
    if (CheckTensorIsVar(tensor)) {
      tensor->set_allocator(context_->allocator);
      tensor->set_data(ins[i++]);
    }
  }
  int j = 0;
  for (auto tensor : kernel->out_tensors()) {
    tensor->set_data(outs[j++]);
  }
}

int DynamicSession::RunKernel(int index, const KernelHook &before, const KernelHook &after) {
  int kernel_num = static_cast<int>(kernels_.size());
  if (index < 0 || index >= kernel_num) {
    MS_LOG(ERROR) << "Specified kernel-index is out-of-side.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (index <= execute_index_) {
    MS_LOG(ERROR) << "Current kernel has executed.";
    return RET_ERROR;
  }
  ++execute_index_;
  while (execute_index_ <= index) {
    MS_CHECK_TRUE_MSG(execute_index_ >= 0, RET_ERROR, "Kernel-index is out-of-side.");
    auto kernel = kernels_[execute_index_];
    if (before) {
      auto res = before(kernel, {kernel->name(), kernel::TypeName(kernel->type())}, sample_num_);
      MS_CHECK_TRUE_MSG(res, RET_ERROR, "Run pre-hook failed.");
    }
    std::vector<void *> in_datas;
    std::vector<void *> out_datas;
    auto ret = PreProcess(kernel, &in_datas, &out_datas);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "PreProcess failed.");
    for (int i = 0; i < sample_num_; ++i) {
      PrepareInOuts(kernel, in_datas, out_datas, i);
      ret = kernel->Execute();
      if (ret != RET_OK) {
        break;
      }
    }
    ResetInOuts(kernel, in_datas, out_datas);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Run kernel failed.";
      return ret;
    }
    if (after) {
      auto res = after(kernel, {kernel->name(), kernel::TypeName(kernel->type())}, sample_num_);
      MS_CHECK_TRUE_MSG(res, RET_ERROR, "Run after-hook failed.");
    }
    for (auto tensor : kernel->in_tensors()) {
      tensor->DecRefCount();
    }
    ++execute_index_;
  }
  if (execute_index_ == static_cast<int>(kernels_.size())) {
    execute_index_ = 0;
  }
  --execute_index_;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
