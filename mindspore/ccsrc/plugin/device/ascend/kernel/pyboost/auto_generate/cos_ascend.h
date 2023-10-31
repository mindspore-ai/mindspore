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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_COS_ASCEND_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_COS_ASCEND_H_

#include "kernel/pyboost/auto_generate/cos.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class CosAscend : public pyboost::Cos {
public:
 CosAscend() = default;
 ~CosAscend() = default;

 tensor::TensorPtr Call(const tensor::TensorPtr &self) override;
};
MS_REG_PYBOOST_OP(Ascend, Cos);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_COS_ASCEND_H_
