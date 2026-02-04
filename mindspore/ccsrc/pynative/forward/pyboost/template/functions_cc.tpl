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

#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include <string>
#include "utils/ms_utils.h"
#include "include/runtime/utils/dispatch/dispatch_env.h"
#include "include/runtime/utils/dispatch/dispatch_env.h"
#include "include/pynative/utils/pyboost/functions/dispatch.h"
#include "include/pynative/utils/pyboost/functions/base.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_reg.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "primitive/auto_generate/gen_ops_def.h"
#include "mindspore/ccsrc/pynative/utils/base.h"
#include "mindspore/ccsrc/pynative/backward/grad_utils.h"
#include "include/pynative/utils/pynative_utils.h"
#include "mindspore/core/include/utils/stream_guard.h"
#include "mindspore/ccsrc/pynative/utils/pyboost/functions/customize/view_impl.h"

${ops_inc}
${pyboost_op_header_include}

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
inline device::DeviceType GetDeviceTarget() { return OpRunStatus::Get().device_target(); }

using TensorPtr = std::shared_ptr<tensor::Tensor>;

CloneFunc inplace_clone_func{nullptr};
}

void RegisterCloneFunc(const CloneFunc &clone_func) {
  inplace_clone_func = clone_func;
}

const CloneFunc& GetCloneFunc() {
  if (inplace_clone_func == nullptr) {
    MS_LOG(EXCEPTION) << "Clone func not register!";
  }
  return inplace_clone_func;
}

${op_call_with_grad}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
