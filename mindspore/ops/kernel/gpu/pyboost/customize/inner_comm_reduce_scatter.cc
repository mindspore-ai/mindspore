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

#include "kernel/gpu/pyboost/customize/inner_comm_reduce_scatter.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/customize/op_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void InnerCommReduceScatterGPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                        const Int64ImmPtr &rank_size, const StringImmPtr &op_type,
                                        const StringImmPtr &group) {
  CommonCommFunc(op, input_tensor, nullptr, nullptr);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
