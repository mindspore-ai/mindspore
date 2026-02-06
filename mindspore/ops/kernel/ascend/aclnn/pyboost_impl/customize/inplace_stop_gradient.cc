/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_stop_gradient.h"
#include <memory>
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/utils/pynative/variable.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void InplaceStopGradientAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "Inplace StopGradient Ascend start";
  if (pynative::autograd::impl::GetViewAutogradMetaImpl(input_tensor) != nullptr) {
    MS_LOG(EXCEPTION) << "Cannot stop_gradient view inplace";
  }
  runtime::Pipeline::Get().WaitBpropStage();
  input_tensor->set_need_pipeline_sync(true);
  input_tensor->set_auto_grad_meta_data(nullptr);
  if (input_tensor->param_info() != nullptr) {
    input_tensor->param_info()->set_requires_grad(false);
  }
  op->set_outputs({input_tensor});
  MS_LOG(DEBUG) << "Inplace StopGradient Ascend end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
