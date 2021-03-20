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

namespace mindspore::lite {

int MindrtExecutor::Prepare(const std::vector<kernel::LiteKernel *> &kernels) {
  auto ret = MindrtInit();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MindrtInit failed";
    return ret;
  }
  auto kernelSize = kernels.size();
  opActors_ = CreateOpActor(kernels);
  if (opActors_.size() != kernelSize) {
    MS_LOG(ERROR) << "CreateOpActor failed";
    return RET_ERROR;
  }
  for (size_t i = 0; i < kernelSize; i++) {
    if (kernels[i]->in_kernels().size() == 0) {
      auto inTensorSize = kernels[i]->in_tensors().size();

      for (size_t j = 0; j < inTensorSize; j++) {
        auto data =
          std::make_shared<OpData<Tensor>>(opActors_[i]->GetAID(), kernels[i]->in_tensors()[j], static_cast<int>(j));
        inputData_.emplace_back(data);
      }
    }

    if (kernels[i]->out_kernels().size() == 0) {
      auto outTensorSize = kernels[i]->out_tensors().size();

      for (size_t j = 0; j < outTensorSize; j++) {
        auto data =
          std::make_shared<OpData<Tensor>>(opActors_[i]->GetAID(), kernels[i]->out_tensors()[j], static_cast<int>(j));
        outputData_.emplace_back(data);
      }
    }
  }
  return RET_OK;
}

int MindrtExecutor::Run(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                        const std::vector<kernel::LiteKernel *> &kernels, mindspore::Allocator *allocator,
                        const KernelCallBack &before, const KernelCallBack &after) {
  MS_ASSERT(nullptr != allocator);
  if (kernels.front()->Type() != schema::PrimitiveType_Merge) {
    auto ret = this->CheckInputs(in_tensors);
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "CheckInputs failed";
      return ret;
    }
  }
  // clear ref_count
  for (auto *kernel : kernels) {
    for (auto *tensor : kernel->in_tensors()) {
      tensor->set_ref_count(0);
    }
  }

  return MindrtRun<Tensor>(inputData_, &outputData_, &before, &after);
}

}  // namespace mindspore::lite
