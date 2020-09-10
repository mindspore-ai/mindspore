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

#include "include/train_session.h"
#include <algorithm>
#include "utils/log_adapter.h"
#include "include/context.h"
#include "src/common/utils.h"
#include "mindspore/lite/src/tensor.h"
#include "src/train/loss_kernel.h"
#include "src/train/train_populate_parameter.h"
#include "src/runtime/runtime_api.h"
#include "src/executor.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32_grad/convolution.h"

namespace mindspore::session {

TrainSession::TrainSession() { kernel::PopulateTrainParameters(); }

void TrainSession::ReplaceOps() {
  mindspore::lite::KernelRegistrar tmp(mindspore::kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32,
                                       mindspore::schema::PrimitiveType_Conv2D,
                                       mindspore::kernel::CpuConvTrainFp32KernelCreator);
}

int TrainSession::CompileGraph(lite::Model *model) {
  model_ = model;
  ReplaceOps();
  return LiteSession::CompileGraph(model);
}

void *TrainSession::ExportToBuf(void *buf, size_t *len) const {
  //  auto train_model_impl = (dynamic_cast<lite::train::TrainModelImpl*>(model_->model_impl()));
  //  return train_model_impl->ExportToBuf(buf, len);
  return nullptr;
}

int TrainSession::RunGraph(const session::KernelCallBack &before, const session::KernelCallBack &after) {
  auto ms_output_tensors = GetOutputMap();
  this->outputs_.clear();
  for (auto ms_tensors : ms_output_tensors)
    for (auto ms_tensor : ms_tensors.second) this->outputs_.push_back((dynamic_cast<lite::Tensor *>(ms_tensor)));
  if (train_mode_) return LiteSession::RunGraph(before, after);

  // object is expected to run only inference part of graph
  // prepare a lit of kernels till the loss function -- temporary solution
  std::vector<kernel::LiteKernel *> infference_kernels;
  for (auto kernel : this->kernels_) {
    if (dynamic_cast<const kernel::LossKernel *>(kernel) != nullptr) break;
    infference_kernels.push_back(kernel);
  }

  MS_EXCEPTION_IF_NULL(this->context_);
  // TODO(Emir)
  // SetMaxWokerNum(context_->thread_num_);
  // context_->running_ = true;
  lite::Executor executor;
  if (before == nullptr && after == nullptr) {
    return executor.Run(this->inputs_, this->outputs_, infference_kernels, this->context_->allocator.get());
  } else {
    return executor.Run(this->inputs_, this->outputs_, infference_kernels, this->context_->allocator.get(), before,
                        after);
  }
}

void TrainSession::train() {
  for (auto *kernel : kernels_) {
    MS_ASSERT(nullptr != kernel);
    kernel->train();
  }
  train_mode_ = true;
  ext_output_map_.clear();
  for (auto kernel : this->kernels_) {
    if (dynamic_cast<const kernel::LossKernel *>(kernel) != nullptr) {
      auto *ms_tensor = new lite::Tensor(*kernel->out_tensors().at(0));
      ext_output_map_[kernel->name()].emplace_back(ms_tensor);
    }
  }
}

void TrainSession::eval() {
  for (auto *kernel : kernels_) {
    MS_ASSERT(nullptr != kernel);
    kernel->eval();
  }
  train_mode_ = false;
  kernel::LiteKernel *last_kernel = nullptr;
  // We should get in_kernels and then get all last kernels
  ext_output_map_ = output_node_map_;
  for (auto kernel : this->kernels_) {
    if ((dynamic_cast<const kernel::LossKernel *>(kernel) != nullptr) && (last_kernel != nullptr)) {
      auto *ms_tensor = new lite::Tensor(*last_kernel->out_tensors().at(0));
      ext_output_map_[last_kernel->name()].emplace_back(ms_tensor);
    }
    last_kernel = kernel;
  }
}

std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> TrainSession::GetOutputMap() const {
  return ext_output_map_;
}
std::vector<tensor::MSTensor *> TrainSession::GetOutputsByName(const std::string &name) const {
  auto ret_vect = LiteSession::GetOutputsByNodeName(name);  // TODO(emir):  GetOutputsByTensorName?
  if (ret_vect.size() > 0) return ret_vect;
  auto ret = ext_output_map_.find(name);
  if (ret == ext_output_map_.end()) {
    MS_LOG(WARNING) << "Node  " << name << " is not an output node";
    std::vector<mindspore::tensor::MSTensor *> empty_ret;
    return empty_ret;
  }
  return ret->second;
}

}  // namespace mindspore::session
