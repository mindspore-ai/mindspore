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
#include "src/sub_graph_kernel.h"

namespace mindspore {
namespace lite {

typedef enum { GRAPH, OP_BY_OP } MindRTMode;

class LiteOpActor : public OpActor<lite::Tensor> {
 public:
  explicit LiteOpActor(kernel::LiteKernel *kernel) : OpActor<lite::Tensor>(kernel->name()), kernel_(kernel) {}
  virtual ~LiteOpActor() = default;
  virtual void OpRun(OpDataPtr<Tensor> inputs, OpContext<Tensor> *context = nullptr) {
    auto op_uuid = context->sequential_num_;
    input_op_datas_[op_uuid].push_back(inputs);
    if (input_op_datas_[op_uuid].size() < kernel_->in_tensors().size()) {
      return;
    }
    auto ret = RunKernel(*(reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_before_)),
                         *(reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_after_)));
    if (ret != RET_OK) {
      input_op_datas_.erase(op_uuid);
      context->SetFailed(ret);
      return;
    }
    input_op_datas_.erase(op_uuid);
    SetOutputData(context);
  }
  void Init() {
    auto ret = CompileArrow();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CompileArrow failed, name: " << kernel_->name();
      // do not support return error
    }
  }
  int CompileArrow();
  int RunKernel(const KernelCallBack &before, const KernelCallBack &after) {
    int ret;
    ret = kernel_->PreProcess();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "PreProcess kernel failed, name: " << kernel_->name();
      return ret;
    }
    ret = kernel_->Run(before, after);
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
void MindrtTerminate(std::vector<std::shared_ptr<LiteOpActor>>);

std::vector<std::shared_ptr<LiteOpActor>> CreateOpActor(const std::vector<kernel::LiteKernel *> &kernels);

}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITE_MINDRT_H_
