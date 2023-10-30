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

#include "kernel/pyboost/op/transpose.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr Transpose::Call(const tensor::TensorPtr &input, const ValueTuplePtr &input_perm) {
  // TODO: kernel_mod->launch
  return mindspore::tensor::TensorPtr();
}
void Transpose::PyboostProcessView(const tensor::TensorPtr &input, const std::vector<int64_t> &input_perm) {
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kAscendDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);
  auto storage_info_list = ops::TransposeCalcDirect(input, input_perm);
  if (!storage_info_list.empty()) {
    storage_info_list[0]->data_type = input->data_type();
    runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, input, "input");
    PyBoostUtils::CreateOutputTensor(input, storage_info_list[0], &outputs_);
    output_abs_ = output(0)->ToAbstract();
  } else {
    MS_LOG_EXCEPTION << "View unsupported:" << primitive_->name();
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
