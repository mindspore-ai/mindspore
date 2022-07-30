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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_MINDRT_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_MINDRT_EXECUTOR_H_

#include <memory>
#include <vector>
#include <unordered_map>
#include <set>
#include <utility>
#include "src/litert/inner_allocator.h"
#include "src/litert/kernel_exec.h"
#include "src/litert/lite_mindrt.h"
#include "src/litert/executor.h"
#include "mindrt/src/actor/actormgr.h"

namespace mindspore::lite {
class MindrtExecutor : public Executor {
 public:
  explicit MindrtExecutor(std::unordered_map<Tensor *, Tensor *> *output_map,
                          std::unordered_map<Tensor *, Tensor *> *input_map)
      : isolate_output_map_(output_map), isolate_input_map_(input_map) {}
  virtual ~MindrtExecutor() { MindrtTerminate(op_actors_, actor_mgr_); }

  int Prepare(const std::vector<kernel::KernelExec *> &kernels, const std::vector<Tensor *> &inputs,
              const std::vector<Tensor *> &outputs, lite::InnerContext *ctx) override;

  int Run(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
          const std::vector<kernel::KernelExec *> &kernels, const KernelCallBack &before = nullptr,
          const KernelCallBack &after = nullptr) override;

  int Resize(const std::vector<mindspore::lite::Tensor *> &inputs, const std::vector<std::vector<int>> &dims) override;

 private:
  int TransferGraphOutput();
  void FreeOutputTensor();
  std::unordered_map<void *, std::set<std::pair<AID, size_t>>> BuildReceiverMap();

 protected:
  int PrepareGraphInput(const std::vector<kernel::KernelExec *> &kernels, const std::vector<Tensor *> &inputs);
  int PrepareGraphOutput(const std::vector<kernel::KernelExec *> &kernels, const std::vector<Tensor *> &outputs);
  int PreInitActors();
  int LinkActors();
  int PostInitActors();
  std::vector<std::shared_ptr<LiteOpActor>> op_actors_;
  std::vector<OpDataPtr<Tensor>> input_data_;
  std::vector<OpDataPtr<Tensor>> output_data_;
  std::unordered_map<Tensor *, Tensor *> *isolate_output_map_;
  std::unordered_map<Tensor *, Tensor *> *isolate_input_map_;
  std::shared_ptr<ActorMgr> actor_mgr_;
};

}  // namespace mindspore::lite
#endif
