/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/parallel_lite_actor.h"
#include <utility>
#include <algorithm>
#include "src/litert/lite_mindrt.h"
#include "mindrt/include/mindrt.hpp"
#include "src/litert/kernel_exec_util.h"
#include "src/common/tensor_util.h"
#include "src/common/common.h"
#include "src/litert/inner_allocator.h"
#include "src/litert/kernel/cpu/base/partial_fusion.h"

namespace mindspore::lite {
void ParallelLiteActor::RunOpData(OpData<lite::Tensor> *inputs, mindspore::OpContext<lite::Tensor> *context) {
  auto op_uuid = context->sequential_num_;
  input_op_datas_[op_uuid].push_back(inputs);
  inputs_data_[inputs->index_] = inputs->data_;
  if (input_op_datas_[op_uuid].size() < kernel_->in_tensors().size()) {
    return;
  }
  auto ret = InitInputData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "run kernel failed, name: " << kernel_->name();
    context->SetFailed(ret);
    return;
  }
  SetOpContext(context);
  auto subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel_);
  if (MS_UNLIKELY(subgraph_kernel->GetGraphChanged())) {
    if (PostInit() != RET_OK) {
      MS_LOG(ERROR) << "KernelActorInit failed, name: " << kernel_->name();
      context->SetFailed(RET_ERROR);
      return;
    }
  }
  if (MS_UNLIKELY(!finish_)) {
    // It is uniformly cleared to prevent the residual count caused by the failure of the last run from affecting this
    // run
    for (auto &kernels_actor : kernels_actors_) {
      kernels_actor->ClearReady();
    }
  }
  finish_ = false;
  output_data_count_ = 0;

  for (size_t i = 0; i < begin_readly_indexs_.size(); i++) {
    Async(kernels_actors_[begin_readly_indexs_[i]]->GetAID(), get_actor_mgr(), &mindspore::lite::KernelsActor::Run);
  }
  return;
}

void ParallelLiteActor::AddOutputDataCount() {
  auto last_count = output_data_count_.fetch_add(1);
  if (static_cast<size_t>(last_count + 1) == output_data_arrows_.size()) {
    input_op_datas_.erase(op_context_->sequential_num_);
    output_data_count_ = 0;
    finish_ = true;
  }
}

void ParallelLiteActor::DelKernelsActors() {
  MS_LOG(INFO) << "start del KernelsActors.";
  for (const auto &actor : kernels_actors_) {
    mindspore::Terminate(actor->GetAID(), get_actor_mgr());
  }
  kernels_actors_.clear();
  MS_LOG(INFO) << "end del KernelsActors.";
}

ParallelLiteActor::~ParallelLiteActor() { DelKernelsActors(); }

int ParallelLiteActor::KernelActorInit() {
  if (results_tensor_index_.size() != results_index_.size()) {
    MS_LOG(ERROR) << "results_tensor_index_ size " << results_tensor_index_.size()
                  << " not equal to results_index_ size" << results_index_.size();
    return RET_ERROR;
  }
  size_t max_tensor_index = kernel_->out_tensors().size();
  MS_ASSERT(
    std::find_if(results_tensor_index_.begin(), results_tensor_index_.end(), [max_tensor_index](const int index) {
      return static_cast<size_t>(index) >= max_tensor_index;
    }) == results_tensor_index_.end());

  auto subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel_);
  kernel::KernelsArray split_kernels;
  auto ret = subgraph_kernel->SubGraphSplitByOperator(&split_kernels);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SubGraphSplitByOperator failed.";
    return ret;
  }
  subgraph_kernel->SetGraphChanged(false);
  size_t units_size = split_kernels.units.size();
  if (units_size == 0) {
    MS_LOG(ERROR) << "split_kernels size is 0.";
    return RET_ERROR;
  }

  if (output_data_arrows_.size() == 0) {
    MS_LOG(ERROR) << "output_data_arrows_ size is 0.";
    return RET_ERROR;
  }
  std::vector<lite::Tensor *> graph_output_tensor;
  for (size_t i = 0; i < output_data_arrows_.size(); i++) {
    auto &arrow = output_data_arrows_[i];
    if (static_cast<size_t>(arrow->from_output_index_) >= max_tensor_index) {
      MS_LOG(ERROR) << "arrow->from_output_index_ " << arrow->from_output_index_ << " greater than tensor maximum "
                    << max_tensor_index;
      return RET_ERROR;
    }
    graph_output_tensor.push_back(kernel_->out_tensors().at(arrow->from_output_index_));
  }
  auto thread_pool = reinterpret_cast<ActorThreadPool *>(ctx_->thread_pool_);
  size_t graph_output_size = graph_output_tensor.size();

  int kernels_actor_num = 0;
  DelKernelsActors();
  std::string kernels_actor_name = "_" + std::to_string(split_kernels.units.size()) + "_" + GetAID().Name();
  for (auto &unit : split_kernels.units) {
    if (unit.kernels.size() == 0) {
      MS_LOG(ERROR) << "kernels size is 0.";
      ret = RET_ERROR;
      break;
    }
    auto kernels_actor = std::make_shared<KernelsActor>(
      this, std::to_string(kernels_actor_num++) + kernels_actor_name + unit.kernels.front()->name(), unit.kernels);
    if (kernels_actor == nullptr) {
      MS_LOG(ERROR) << "new kernels_actor failed.";
      ret = RET_ERROR;
      break;
    }
    for (auto &kernel : unit.kernels) {
      for (auto &tensor : kernel->out_tensors()) {
        for (size_t i = 0; i < graph_output_size; i++) {
          if (graph_output_tensor[i] == tensor) {
            kernels_actor->SetHaveOutput(true);
            kernels_actor->AddOutputDataArrows(output_data_arrows_[i]);
            kernels_actor->AddOutputData(outputs_data_[i]);
            for (size_t j = 0; j < results_tensor_index_.size(); j++) {
              if (results_tensor_index_[j] == static_cast<size_t>(output_data_arrows_[i]->from_output_index_)) {
                kernels_actor->AddResultsIndex(results_index_[j]);
              }
            }
          }
        }
      }
    }
    kernels_actor->SetInActorIndexs(unit.input_indexs);
    kernels_actor->SetOutActorIndexs(unit.output_indexs);
    kernels_actor->SetIsSignleIn(unit.input_indexs.size() <= 1);
    kernels_actor->set_thread_pool(thread_pool);
    kernels_actor->set_actor_mgr(get_actor_mgr());
    this->AddKernelsActor(kernels_actor);
    (void)mindspore::Spawn(kernels_actor);
  }
  if (ret != RET_OK) {
    kernels_actors_.clear();
    this->SetBeginReadlyIndexs({});
  } else {
    this->SetBeginReadlyIndexs(split_kernels.graph_input);
  }

  return ret;
}

int ParallelLiteActor::PostInit() {
  auto ret = PrepareOutputData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "run PrepareOutputData failed, name: " << kernel_->name();
    return ret;
  }
  ret = KernelActorInit();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "run KernelActorInit failed, name: " << kernel_->name();
    return ret;
  }
  return RET_OK;
}

void ParallelLiteActor::CheckReadyActors(const std::vector<size_t> &indices) {
  for (size_t i = 0; i < indices.size(); i++) {
    if (kernels_actors_[indices[i]]->GetReady()) {
      Async(kernels_actors_[indices[i]]->GetAID(), get_actor_mgr(), &mindspore::lite::KernelsActor::Run);
    }
  }
}

void KernelsActor::Run() {
  mindspore::OpContext<lite::Tensor> *context = parallel_lite_actor_->OpContext();
  const KernelCallBack &before = *reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_before_);
  const KernelCallBack &after = *reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_after_);

  for (auto &kernel : nodes_) {
    MS_ASSERT(kernel != nullptr);
    auto ret = kernel->Execute(before, after);
    if (MS_UNLIKELY(ret != RET_OK)) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
      context->SetFailed(ret);
      return;
    }
  }
  parallel_lite_actor_->CheckReadyActors(out_actors_indexs_);
  if (have_output_) {
    auto output_size = output_data_arrows_.size();
    for (size_t i = 0; i < output_size; ++i) {
      auto data = outputs_data_[i];
      Async(output_data_arrows_[i]->to_op_id_, get_actor_mgr(), &mindspore::OpActor<Tensor>::RunOpData, data.get(),
            context);
      parallel_lite_actor_->AddOutputDataCount();
    }
    for (auto &index : results_index_) {
      context->SetResult(index, RET_OK);
    }
  }
}
}  // namespace mindspore::lite
