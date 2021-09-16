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
#include <queue>
#include <memory>
#include "src/mindrt_executor.h"
#include "src/lite_mindrt.h"
#include "include/errorcode.h"
#include "src/common/tensor_util.h"
#include "nnacl/base/cast_base.h"

namespace mindspore::lite {
int MindrtExecutor::PrepareInputData(const std::vector<kernel::LiteKernel *> &kernels,
                                     const std::vector<Tensor *> &inputs) {
  for (size_t j = 0; j < kernels.size(); ++j) {
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
      if (data == nullptr) {
        MS_LOG(ERROR) << "new opdata failed.";
        return RET_NULL_PTR;
      }
      input_data_.emplace_back(data);
    }
  }
  return RET_OK;
}

int MindrtExecutor::PrepareOutputData(const std::vector<kernel::LiteKernel *> &kernels,
                                      const std::vector<Tensor *> &outputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    Tensor *graph_output_tensor = outputs[i];
    if (graph_output_tensor->IsGraphInput()) {
      continue;
    }
    auto current_output_map =
      std::find_if(output_tensor_map_->begin(), output_tensor_map_->end(), [&](const auto output_map_tensor) {
        if (graph_output_tensor == output_map_tensor.second) {
          return true;
        }
        return false;
      });
    MS_ASSERT(current_output_map != output_tensor_map_->end());
    Tensor *subgraph_output_tensor = current_output_map->first;

    for (size_t j = 0; j < kernels.size(); ++j) {
      auto out_tensor_size = kernels[j]->out_tensors().size();
      for (size_t k = 0; k < out_tensor_size; ++k) {
        if (subgraph_output_tensor != kernels[j]->out_tensors()[k]) {
          continue;
        }
        auto data =
          std::make_shared<OpData<Tensor>>(op_actors_[j]->GetAID(), subgraph_output_tensor, static_cast<int>(k));
        if (data == nullptr) {
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

int MindrtExecutor::Prepare(const std::vector<kernel::LiteKernel *> &kernels, const std::vector<Tensor *> &inputs,
                            const std::vector<Tensor *> &outputs, const lite::InnerContext *ctx) {
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

  ret = PrepareInputData(kernels, inputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PrepareInputData failed";
    return ret;
  }

  ret = PrepareOutputData(kernels, outputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PrepareOutputData failed";
    return ret;
  }

  for (auto actor : op_actors_) {
    ret = actor->LiteActorInit(&op_actors_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "LiteActorInit failed, actor aid: " << actor->GetAID();
      return ret;
    }
  }

  return RET_OK;
}

void MindrtExecutor::TransferGraphOutput() {
  for (auto tensor_map : *output_tensor_map_) {
    auto dst_tensor = tensor_map.second;
    auto src_tensor = tensor_map.first;
    dst_tensor->set_shape(src_tensor->shape());
    /* dst tensor free in FreeOutputTensor */

    if (src_tensor->data_type() == kNumberTypeFloat16) {
      dst_tensor->MallocData();
      Fp16ToFloat32(reinterpret_cast<uint16_t *>(src_tensor->MutableData()),
                    reinterpret_cast<float *>(dst_tensor->data()), dst_tensor->ElementsNum());
    } else {
      dst_tensor->set_data(src_tensor->data());
      src_tensor->set_data(nullptr);
    }
    src_tensor->DecRefCount();
  }
  return;
}

void MindrtExecutor::FreeOutputTensor() {
  for (auto tensor_map : *output_tensor_map_) {
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
