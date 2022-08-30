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

#include "src/litert/cxx_api/kernel_executor/kernel_executor.h"
#include "src/litert/cxx_api/kernel_executor/kernel_executor_impl.h"

namespace mindspore {
Status KernelExecutor::Build(const std::shared_ptr<ops::BaseOperator> &op, const std::vector<MSTensor> &inputs,
                             const std::shared_ptr<Context> &ms_context) {
  if (impl_ == nullptr) {
    impl_ = std::make_shared<KernelExecutorImpl>();
    if (impl_ == nullptr) {
      MS_LOG(ERROR) << "implement is null.";
      return kLiteNullptr;
    }
  }

  return impl_->Build(op, inputs, ms_context);
}

Status KernelExecutor::Build(const std::shared_ptr<ops::Custom> &op, const std::vector<MSTensor> &inputs,
                             const std::shared_ptr<Context> &ms_context, const int output_num) {
  if (impl_ == nullptr) {
    impl_ = std::make_shared<KernelExecutorImpl>();
    if (impl_ == nullptr) {
      MS_LOG(ERROR) << "implement is null.";
      return kLiteNullptr;
    }
  }

  return impl_->Build(op, inputs, ms_context, output_num);
}

Status KernelExecutor::ReSize(const std::vector<MSTensor> &inputs) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "implement is null.";
    return kLiteNullptr;
  }
  return impl_->ReSize(inputs);
}

Status KernelExecutor::Execute(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "implement is null.";
    return kLiteNullptr;
  }
  return impl_->Execute(inputs, outputs);
}
}  // namespace mindspore
