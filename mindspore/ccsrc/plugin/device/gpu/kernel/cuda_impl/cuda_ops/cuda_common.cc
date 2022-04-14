/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace gpu {
CudaCommon &CudaCommon::GetInstance() {
  static CudaCommon instance;
  return instance;
}

CudaCommon::CudaCommon() { device_id_ = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID); }
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
