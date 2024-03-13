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
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"

namespace mindspore {
namespace kernel {
MS_ACLNN_COMMON_KERNEL_FACTORY_REG(LogicalXor, aclnnLogicalXor, 3)
MS_ACLNN_COMMON_KERNEL_FACTORY_REG(Maximum, aclnnMaximum, 3)
MS_ACLNN_COMMON_KERNEL_FACTORY_REG(Minimum, aclnnMinimum, 3)
MS_ACLNN_COMMON_KERNEL_FACTORY_REG(NotEqual, aclnnNeTensor, 3)
MS_ACLNN_COMMON_KERNEL_FACTORY_REG(RealDiv, aclnnDiv, 3)
MS_ACLNN_COMMON_KERNEL_FACTORY_REG(Rsqrt, aclnnRsqrt, 2)
}  // namespace kernel
}  // namespace mindspore
