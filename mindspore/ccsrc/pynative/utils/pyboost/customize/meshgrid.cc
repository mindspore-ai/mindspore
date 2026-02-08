/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "include/pynative/utils/pyboost/customize/meshgrid.h"
#include <memory>
#include <utility>
#include <string>
#include "include/backend/common/device_address_utils.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "ops/op_def_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::TensorPtr> MeshgridCustomizeCall(const std::shared_ptr<OpRunner> &op,
                                                     const ValueTuplePtr &tensors_list, const int64_t &indexing_imm) {
  MS_LOG(DEBUG) << "Meshgrid call start";
  RequireGradGuard requires_grad_guard{false};
  auto outputs = meshgrid(tensors_list, indexing_imm);
  op->set_outputs(outputs);
  MS_LOG(DEBUG) << "Meshgrid call end";
  return outputs;
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
