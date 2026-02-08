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

#include "include/pynative/utils/pyboost/customize/cell_backward_hook.h"
#include <memory>
#include "ir/tensor.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/utils/pynative/variable.h"

namespace mindspore::kernel::pyboost {
void CellBackwardHookCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &tensors_list) {
  MS_LOG(DEBUG) << "Cell BackwardHook call start";
  const auto tensor_vector = ConvertValueTupleToVector<tensor::TensorPtr>(tensors_list);
  tensor::TensorPtrList outputs;
  outputs.reserve(tensors_list->size());
  const bool is_multi_output = tensors_list->size() > 1;
  const bool requires_grad = OpRunStatus::Get().RequireGrad();
  for (const auto &tensor : tensor_vector) {
    TensorPtr view_tensor;
    {
      RequireGradGuard require_grad_guard(false);
      view_tensor = view(tensor, tensor->shape());
    }
    auto view_auto_grad_meta_data = pynative::autograd::impl::GetViewAutogradMetaImpl(view_tensor);
    MS_EXCEPTION_IF_NULL(view_auto_grad_meta_data);

    if (requires_grad) {
      view_auto_grad_meta_data->set_creation_type(is_multi_output ? pynative::autograd::CreationType::kMultiOutput
                                                                  : pynative::autograd::CreationType::kCustomBprop);
    }

    (void)outputs.emplace_back(view_tensor);
  }

  op->set_outputs(outputs);
  MS_LOG(DEBUG) << "Cell BackwardHook call end";
}
}  // namespace mindspore::kernel::pyboost
