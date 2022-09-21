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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_DELEGATE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_DELEGATE_H_

#include <vector>
#include <map>
#include "include/api/delegate.h"
#include "src/litert/delegate/npu/npu_manager.h"
#include "src/litert/delegate/npu/pass/npu_pass_manager.h"
#include "src/litert/delegate/npu/op/npu_op.h"
#include "src/litert/inner_context.h"

namespace mindspore::lite {
class NPUDelegate : public Delegate {
 public:
  explicit NPUDelegate(NpuDeviceInfo device_info) : Delegate() { frequency_ = device_info.frequency_; }

  ~NPUDelegate() override;

  Status Init() override;

  Status Build(DelegateModel<schema::Primitive> *model) override;

 protected:
  NPUOp *GetOP(kernel::Kernel *kernel, const schema::Primitive *primitive);

  kernel::Kernel *CreateNPUGraph(const std::vector<NPUOp *> &ops, DelegateModel<schema::Primitive> *model,
                                 KernelIter from, KernelIter end);

  Status AddPasses();

  NPUManager *npu_manager_ = nullptr;
  NPUPassManager *pass_manager_ = nullptr;
  std::map<schema::PrimitiveType, NPUGetOp> op_func_lists_;
  int frequency_ = 0;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_DELEGATE_H_
