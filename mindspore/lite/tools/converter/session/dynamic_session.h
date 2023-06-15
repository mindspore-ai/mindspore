/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_SESSION_DYNAMIC_SESSION_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_SESSION_DYNAMIC_SESSION_H_
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "src/litert/inner_context.h"
#include "src/litert/lite_session.h"
#include "src/executor/kernel_exec.h"
#include "src/tensor.h"
#include "schema/model_generated.h"

namespace mindspore {
namespace lite {
using KernelHook = std::function<bool(kernel::KernelExec *, const MSCallBackParam &, int sample_num)>;

class DynamicSession : public LiteSession {
 public:
  explicit DynamicSession(const std::shared_ptr<InnerContext> &context) { context_ = context; }
  ~DynamicSession() override = default;
  int CompileGraph(Model *model) override;
  int RunKernel(int index, const KernelHook &before = nullptr, const KernelHook &after = nullptr);
  std::vector<Tensor *> GetInTensors(int index);
  std::vector<Tensor *> GetOutTensors(int index);
  size_t GetKernelNum() { return kernels_.size(); }
  void SetSampleNum(int sample_num) { sample_num_ = sample_num; }
  void SetWeightsName(const std::set<std::string> &weights_name) { weights_name_ = weights_name; }

 private:
  int PreProcess(kernel::KernelExec *, std::vector<void *> *, std::vector<void *> *);
  void PrepareInOuts(kernel::KernelExec *, const std::vector<void *> &, const std::vector<void *> &, int);
  void ResetInOuts(kernel::KernelExec *, const std::vector<void *> &, const std::vector<void *> &);
  bool CheckTensorIsVar(Tensor *);
  int sample_num_{0};
  int execute_index_{-1};
  std::set<std::string> weights_name_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_SESSION_DYNAMIC_SESSION_H_
