/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/struct_infer/structure_new_params.h"
#include "utils/log_adapter.h"

namespace mindspore::kernel::pyboost::op_infer {
int64_t npu_gelu_approximate_mode(const std::string &approximate) {
  MS_EXCEPTION_IF_CHECK_FAIL(approximate == "tanh" || approximate == "none",
                             "approximate argument must be either none or tanh.");
  int64_t approximate_mode = approximate == "tanh" ? 1 : 0;
  return approximate_mode;
}
}  // namespace mindspore::kernel::pyboost::op_infer
