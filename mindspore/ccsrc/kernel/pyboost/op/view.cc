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

#include "kernel/pyboost/op/view.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "ops/view/view_strides_calc.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void View::PyboostProcessView(const tensor::TensorPtr &input, const std::vector<int64_t> &shape,
                              const std::string &device_target) {
  MS_EXCEPTION_IF_NULL(input);

  auto ori_storage_info = input->storage_info();
  if (ori_storage_info != nullptr && !ori_storage_info->is_contiguous) {
    MS_LOG(EXCEPTION) << "input tensor:" << input->ToString()
                      << " is not contiguous, storage info:" << ori_storage_info->ToString();
  }

  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_target, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);
  auto storage_info_list = ops::ViewCalcImpl(primitive_, input, shape);
  if (!storage_info_list.empty()) {
    storage_info_list[0]->data_type = input->data_type();
    runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, input, "input");
    PyBoostUtils::CreateOutputTensor(input, storage_info_list[0], &outputs_);
    output_abs_ = output(0)->ToAbstract();
  } else {
    MS_LOG_EXCEPTION << "View unsupported:" << primitive_->name();
  }
}

tensor::TensorPtr View::Call(const tensor::TensorPtr &input, const ValueTuplePtr &shape) {
  // TODO: kernel_mod->launch
  return mindspore::tensor::TensorPtr();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
