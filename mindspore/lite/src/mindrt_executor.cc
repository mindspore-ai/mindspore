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
#include "src/mindrt_executor.h"
#include <queue>
#include <memory>
#include "src/lite_mindrt.h"
#include "include/errorcode.h"
#include "src/common/common.h"
#include "src/common/tensor_util.h"
#ifdef ENABLE_FP16
#include "nnacl/base/cast_base.h"
#endif
#include "src/lite_kernel_util.h"

namespace mindspore::lite {
int MindrtExecutor::PrepareGraphInput(const std::vector<kernel::LiteKernel *> &kernels,
                                      const std::vector<Tensor *> &inputs) {
  auto kernels_size = kernels.size();
  for (size_t j = 0; j < kernels_size; ++j) {
    auto in_tensor_size = kernels[j]->in_tensors().size();
    for (size_t k = 0; k < in_tensor_size; ++k) {
      auto tensor = kernels[j]->in_tensors()[k];
      if (!tensor->IsGraphInput()) {
        continue;
      }
      size_t idx = std::find(inputs.begin(), inputs.end(), tensor) - inputs.begin();
      if (idx == inputs.size()) {
        MS_LOG(ERROR) << "The input is not found.";
        return RET_ERROR;
      }
      auto data = std::make_shared<OpData<Tensor>>(op_actors_[j]->GetAID(), inputs.at(idx), static_cast<int>(k));
      if (MS_UNLIKELY(data == nullptr)) {
        MS_LOG(ERROR) << "new opdata failed.";
        return RET_NULL_PTR;
      }
      input_data_.emplace_back(data);
    }
  }
  return RET_OK;
}

int MindrtExecutor::PrepareGraphOutput(const std::vector<kernel::LiteKernel *> &kernels,
                                       const std::vector<Tensor *> &outputs) {
  auto outputs_size = outputs.size();
  for (size_t i = 0; i < outputs_size; ++i) {
    Tensor *graph_output_tensor = outputs[i];
    if (graph_output_tensor->IsGraphInput()) {
      continue;
    }
    auto current_output_map =
      std::find_if(isolate_output_map_->begin(), isolate_output_map_->end(), [&](const auto output_map_tensor) {
        if (graph_output_tensor == output_map_tensor.second) {
          return true;
        }
        return false;
      });
    MS_ASSERT(current_output_map != isolate_output_map_->end());
    Tensor *subgraph_output_tensor = current_output_map->first;
    auto kernels_size = kernels.size();
    for (size_t j = 0; j < kernels_size; ++j) {
      auto out_tensor_size = kernels[j]->out_tensors().size();
      for (size_t k = 0; k < out_tensor_size; ++k) {
        if (subgraph_output_tensor != kernels[j]->out_tensors()[k]) {
          continue;
        }
        auto data =
          std::make_shared<OpData<Tensor>>(op_actors_[j]->GetAID(), subgraph_output_tensor, static_cast<int>(k));
        if (MS_UNLIKELY(data == nullptr)) {
          MS_LOG(ERROR) << "new opdata failed.";
          return RET_NULL_PTR;
        }
        op_actors_[j]->AddResultIndex(output_data_.size());
        output_data_.emplace_back(data);
      }
    }
  }
  return RET_OK;
}

int MindrtExecutor::Resize(const std::vector<mindspore::tensor::MSTensor *> &inputs,
                           const std::vector<std::vector<int>> &dims) {
  for (auto actor : op_actors_) {
    actor->ResizeGraphInput(inputs, dims);
  }
  return RET_OK;
}

int MindrtExecutor::PreInitActors() {
  for (auto actor : op_actors_) {
    int ret = actor->PreInit(&op_actors_, isolate_input_map_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "IsolateInputData failed, actor aid: " << actor->GetAID();
      return ret;
    }
  }
  return RET_OK;
}

std::unordered_map<void *, std::set<std::pair<AID, size_t>>> MindrtExecutor::BuildReceiverMap() {
  std::unordered_map<void *, std::set<std::pair<AID, size_t>>> receivers_map{};

  for (auto op_actor : op_actors_) {
    auto input_tensors = op_actor->GetKernel()->in_tensors();
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      auto key = input_tensors[i];
      auto pair = std::make_pair(op_actor->GetAID(), i);
      auto iter = receivers_map.find(key);
      if (iter != receivers_map.end()) {
        iter->second.emplace(pair);
      } else {
        std::set<std::pair<AID, size_t>> tmp_set{pair};
        receivers_map[input_tensors[i]] = tmp_set;
      }
    }
  }
  return receivers_map;
}

int MindrtExecutor::LinkActors() {
  auto receivers_map = BuildReceiverMap();
  for (auto &&op_actor : op_actors_) {
    auto ret = op_actor->CompileArrow(receivers_map);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "actor: " << op_actor->GetAID() << " compile arrow failed.";
      return ret;
    }
  }
  return RET_OK;
}

int MindrtExecutor::PostInitActors() {
  for (auto &&actor : op_actors_) {
    auto ret = actor->PostInit();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "PrepareGraphOutput failed, actor aid: " << actor->GetAID();
      return ret;
    }
  }
  return RET_OK;
}

int MindrtExecutor::Prepare(const std::vector<kernel::LiteKernel *> &kernels, const std::vector<Tensor *> &inputs,
                            const std::vector<Tensor *> &outputs, lite::InnerContext *ctx) {
  MS_ASSERT(ctx != nullptr);
  ctx_ = ctx;
  auto ret = MindrtInit();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MindrtInit failed";
    return ret;
  }
  op_actors_ = CreateOpActor(kernels, ctx);
  if (op_actors_.size() != kernels.size()) {
    MS_LOG(ERROR) << "CreateOpActor failed";
    return RET_ERROR;
  }

  ret = PrepareGraphInput(kernels, inputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PrepareGraphInput failed";
    return ret;
  }

  ret = PrepareGraphOutput(kernels, outputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PrepareGraphOutput failed";
    return ret;
  }

  ret = PreInitActors();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PreInitActors failed";
    return ret;
  }

  ret = LinkActors();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LinkActors failed";
    return ret;
  }

  ret = PostInitActors();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PostInitActors failed";
    return ret;
  }
  return RET_OK;
}

void MindrtExecutor::TransferGraphOutput() {
  for (auto tensor_map : *isolate_output_map_) {
    auto dst_tensor = tensor_map.second;
    auto src_tensor = tensor_map.first;
    dst_tensor->set_shape(src_tensor->shape());
    /* dst tensor free in FreeOutputTensor */
#ifdef ENABLE_FP16
    if (src_tensor->data_type() == kNumberTypeFloat16) {
      dst_tensor->MallocData();
      Fp16ToFloat32(reinterpret_cast<float16_t *>(src_tensor->MutableData()),
                    reinterpret_cast<float *>(dst_tensor->data()), dst_tensor->ElementsNum());
    } else {
#endif
      if (dst_tensor->allocator() != src_tensor->allocator()) {
        dst_tensor->set_allocator(src_tensor->allocator());
      }
      dst_tensor->set_data(src_tensor->data());
      if (IS_RUNTIME_ALLOCATOR(src_tensor->allocator()) == false) {
        src_tensor->set_data(nullptr);
      }
#ifdef ENABLE_FP16
    }
#endif
    src_tensor->DecRefCount();
  }
  return;
}

void MindrtExecutor::FreeOutputTensor() {
  for (auto tensor_map : *isolate_output_map_) {
    auto src_tensor = tensor_map.first;
    auto dst_tensor = tensor_map.second;
    if (dst_tensor->allocator() != nullptr) {
      dst_tensor->FreeData();
    } else {
      if (dst_tensor->data_type() == src_tensor->data_type()) {
        /* user set graph-output-tensor from outside */
        src_tensor->set_data(dst_tensor->data());
        src_tensor->set_own_data(false);
        src_tensor->set_allocator(nullptr);
      }
    }
  }
  return;
}

int MindrtExecutor::Run(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                        const std::vector<kernel::LiteKernel *> &kernels, const KernelCallBack &before,
                        const KernelCallBack &after) {
  CHECK_NULL_RETURN(ctx_);
  auto thread_pool = ctx_->thread_pool();
  CHECK_NULL_RETURN(thread_pool);
  if (ctx_->delegate == nullptr) {
    thread_pool->SetSpinCountMaxValue();
  }

  FreeOutputTensor();

  auto ret = MindrtRun<Tensor>(input_data_, &output_data_, &before, &after);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MindrtRun failed";
    return ret;
  }

  TransferGraphOutput();

  thread_pool->SetSpinCountMinValue();
  return RET_OK;
}
}  // namespace mindspore::lite
