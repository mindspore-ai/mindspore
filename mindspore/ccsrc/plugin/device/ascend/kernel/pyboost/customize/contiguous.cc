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

#include "plugin/device/ascend/kernel/pyboost/customize/contiguous.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "kernel/pyboost/auto_generate/copy.h"
#include "kernel/pyboost/customize/op_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr ContiguousAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                const BaseTensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "Call start";
  MS_EXCEPTION_IF_NULL(input_tensor);

  auto output_tensor = ContiguousTensorOpProcess(op, input_tensor);
  if (output_tensor != nullptr) {
    return output_tensor;
  }

  auto copy_op = CREATE_PYBOOST_OP(Copy, kAscendDevice);
  copy_op->set_stream_id(op->stream_id());
  output_tensor = copy_op->Call(input_tensor);
  op->set_outputs(copy_op->outputs());
  return output_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
