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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_COREML_DELEGATE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_COREML_DELEGATE_H_

#include <vector>
#include <map>
#include "include/api/delegate.h"
#include "src/litert/delegate/coreml/op/coreml_op.h"
#include "src/litert/delegate/coreml/pass/coreml_pass_manager.h"

namespace mindspore {
namespace lite {
class CoreMLDelegateImpl : public mindspore::Delegate {
 public:
  CoreMLDelegateImpl() = default;

  ~CoreMLDelegateImpl() override;

  bool IsSupportCoreML() const;

  Status Init() override;

  Status Build(DelegateModel<schema::Primitive> *model) override;

 protected:
  CoreMLOp *GetOP(kernel::Kernel *kernel, const schema::Primitive *primitive);

  kernel::Kernel *CreateCoreMLGraph(const std::vector<CoreMLOp *> &ops, DelegateModel<schema::Primitive> *model,
                                    KernelIter from, KernelIter end);

  Status AddPasses();

 protected:
  int graph_index_ = 0;
  CoreMLPassManager *pass_manager_ = nullptr;
  std::map<schema::PrimitiveType, CoreMLGetOp> op_func_lists_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_COREML_DELEGATE_H_
