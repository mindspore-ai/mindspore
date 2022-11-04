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

#include "infer/dpico_common_infer.h"
#include <vector>
#include <memory>
#include "utils/log_adapter.h"
#include "common/infer_util.h"
#include "include/errorcode.h"
#include "include/registry/register_kernel_interface.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
namespace kernel {
std::shared_ptr<KernelInterface> DpicoCommonInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<DpicoCommonInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "make shared failed, infer is nullptr.";
    return nullptr;
  }
  return infer;
}
Status DpicoCommonInterface::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                                   const schema::Primitive *primitive, const kernel::Kernel *kernel) {
  auto status = dpico::CheckCustomInputOutput(inputs, outputs, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Check custom input output failed.";
    return kLiteError;
  }
  auto param = primitive->value_as_Custom();
  if (param == nullptr) {
    MS_LOG(ERROR) << "param is nullptr";
    return kLiteError;
  }
  if (param->type() == nullptr) {
    MS_LOG(ERROR) << "param type is nullptr";
    return kLiteError;
  }

  const auto &input = (*inputs)[0];
  auto &output = (*outputs)[0];
  output.SetShape(input.Shape());
  return kSuccess;
}
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Normalize, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Threshold, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, X_LOG_Y, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, X_DIV_Y, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Bnll, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, ShuffleChannel, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Mvn, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Nop, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Bias, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Eltwise, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Hardmax, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Mod, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Shrink, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, BitShift, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Acos, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Acosh, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Asinh, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Atanh, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Cosh, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Sinh, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, HardSigmoid, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Softsign, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Xor, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Mish, DpicoCommonInferCreater)
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Custom, DpicoCommonInferCreater)
}  // namespace kernel
}  // namespace mindspore
