/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_LITE_MINDRT_H_
#define MINDSPORE_LITE_SRC_LITE_MINDRT_H_
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "actor/op_actor.h"
#include "src/lite_kernel.h"
#include "actor/actor.h"
#include "async/uuid_base.h"
#include "async/future.h"

namespace mindspore {
namespace lite {

typedef enum { GRAPH, OP_BY_OP } MindRTMode;

class LiteOpActor : public OpActor<lite::Tensor> {
 public:
  explicit LiteOpActor(kernel::LiteKernel *kernel) : OpActor<lite::Tensor>(kernel->name()), kernel_(kernel) {}
  virtual ~LiteOpActor() = default;
  virtual void OpRun(OpDataPtr<Tensor> inputs, OpContext<Tensor> *context = nullptr) {
    input_op_datas_[context->sequential_num_].push_back(inputs);
    if (input_op_datas_[context->sequential_num_].size() < kernel_->in_tensors().size()) {
      return;
    }
    auto ret = RunKernel();
    if (ret != RET_OK) {
      context->SetFailed(ret);
      input_op_datas_.erase(context->sequential_num_);
      return;
    }
    SetOutputData(context);
    input_op_datas_.erase(context->sequential_num_);
  }
  void Init() {
    auto ret = CompileArrow();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CompileArrow failed, name: " << kernel_->name();
      // do not support return error
    }
  }
  int CompileArrow();
  int RunKernel() {
    int ret;
    ret = kernel_->PreProcess();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "PreProcess kernel failed, name: " << kernel_->name();
      return ret;
    }
    ret = kernel_->Run();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel_->name();
      return ret;
    }
    ret = kernel_->PostProcess();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "PostProcess kernel failed, name: " << kernel_->name();
      return ret;
    }

    return ret;
  }

 private:
  void SetOutputData(OpContext<Tensor> *context);

  kernel::LiteKernel *kernel_;
};

int MindrtInit();
std::vector<std::shared_ptr<LiteOpActor>> CreateOpActor(const std::vector<kernel::LiteKernel *> &kernels);

}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITE_MINDRT_H_
