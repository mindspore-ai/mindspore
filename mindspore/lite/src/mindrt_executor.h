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

#ifndef MINDSPORE_LITE_SRC_MINDRT_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_MINDRT_EXECUTOR_H_
#include <memory>
#include <vector>
#include "src/runtime/allocator.h"
#include "src/lite_kernel.h"
#include "src/lite_mindrt.h"
#include "src/executor.h"
#include "include/lite_session.h"

namespace mindspore::lite {
// class Executor {
class MindrtExecutor : public Executor {
 public:
  MindrtExecutor() = default;
  virtual ~MindrtExecutor() { MindrtTerminate(opActors_); }

  virtual int Prepare(const std::vector<kernel::LiteKernel *> &kernels);

  virtual int Run(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const std::vector<kernel::LiteKernel *> &kernels, mindspore::Allocator *allocator = nullptr,
                  const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr);

 protected:
  std::vector<std::shared_ptr<LiteOpActor>> opActors_;
  std::vector<OpDataPtr<Tensor>> inputData_;
  std::vector<OpDataPtr<Tensor>> outputData_;
};

}  // namespace mindspore::lite
#endif
