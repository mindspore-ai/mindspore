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

#include "plugin/device/ascend/kernel/pyboost/square_ascend.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_mod.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
bool SquareAscend::Launch(const tensor::TensorPtr &input, const tensor::TensorPtr &output) {
  // TODO 使用pow实现square，如下代码做性能验证
  GilReleaseWithCheck release_gil;

  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kAscendDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);

  auto input_address = runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, input, "input");
  auto output_address = runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, output, "output");

  MS_EXCEPTION_IF_NULL(input_address);
  MS_EXCEPTION_IF_NULL(output_address);
  transform::AclRunner runner;

  runner.SetName("Square");
  auto [input_acl_desc, input_acl_data] =
    transform::AclConverter::CreateTensorDesc(input, input->shape(), input_address->format(), "input");
  auto [output_acl_desc, output_acl_data] =
    transform::AclConverter::CreateTensorDesc(output, output->shape(), output_address->format(), "output");

  runner.ResizeOpInputs(1);
  runner.ResizeOpOutputs(1);

  runner.SetInput(0, input_acl_desc, input_acl_data);
  runner.SetOutput(0, output_acl_desc, output_acl_data);

  MS_LOG(DEBUG) << "Begin launch kernel: Square";
  auto stream_ptr = device::ascend::AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  runner.Run(stream_ptr, false);
  MS_LOG(DEBUG) << "End launch kernel: Square";

  return true;
}

bool SquareAscend::LaunchByKernel(const tensor::TensorPtr &input, const tensor::TensorPtr &output) {
  // TODO 使用pow实现square，如下代码做性能验证
  GilReleaseWithCheck release_gil;

  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kAscendDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);

  auto input_address = runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, input, "input");
  auto output_address = runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, output, "output");

  MS_EXCEPTION_IF_NULL(input_address);
  MS_EXCEPTION_IF_NULL(output_address);

  std::vector<std::string> input_device_formats = {input_address->format()};
  auto output_device_formats = {output_address->format()};
  auto input_device_types = {input_address->type_id()};
  auto output_device_types = {output_address->type_id()};

  kernel::AclKernelModPtr square_kernel = std::make_shared<kernel::AclKernelMod>();
  square_kernel->SetPrimitive(primitive_);
  square_kernel->CreateAclConverter();

  square_kernel->SetDeviceInfo(input_device_formats, output_device_formats, input_device_types, output_device_types);
  ShapeVector input_x_shape = input->shape();
  square_kernel->PackageInput(kIndex0, "", &input_x_shape);
  square_kernel->PackageOutput(kIndex0, output->shape());

  auto x_addr = std::make_shared<kernel::Address>(input_address->GetMutablePtr(), input_address->GetSize());
  auto out_addr = std::make_shared<kernel::Address>(output_address->GetMutablePtr(), output_address->GetSize());

  auto stream_ptr = device::ascend::AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  MS_LOG(DEBUG) << "Begin launch kernel: Square";
  auto ret = square_kernel->Launch({x_addr}, std::vector<AddressPtr>{}, {out_addr}, stream_ptr);
  MS_LOG(DEBUG) << "End launch kernel: Square";

  return ret;
}

tensor::TensorPtr SquareAscend::Call(const tensor::TensorPtr &input) {
  InferOutput(input);
  Launch(input, output(0));
  return output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
