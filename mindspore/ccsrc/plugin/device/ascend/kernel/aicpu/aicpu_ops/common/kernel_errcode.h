/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef AICPU_OPS_AICPU_COMMON_KENERL_ERRCODE_H_
#define AICPU_OPS_AICPU_COMMON_KENERL_ERRCODE_H_

namespace aicpu {
enum AicpuKernelErrCode {
  // 0-3 is fixed error code, runtime need interpret 0-3 error codes
  AICPU_KERNEL_STATE_SUCCESS = 0,
  AICPU_KERNEL_STATE_PARAM_INVALID = 1,
  AICPU_KERNEL_STATE_FAILED = 2,
  AICPU_KERNEL_STATE_EXECUTE_TIMEOUT = 3,
  AICPU_KERNEL_STATE_INTERNAL_ERROR = 4,
  AICPU_KERNEL_STATE_END_OF_SEQUENCE = 201,
};
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_COMMON_KENERL_ERRCODE_H_
