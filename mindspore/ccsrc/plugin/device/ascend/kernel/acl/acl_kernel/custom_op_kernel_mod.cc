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
#include "plugin/device/ascend/kernel/acl/acl_kernel/custom_op_kernel_mod.h"
#include <memory>
#include <string>
#include <algorithm>
#include <functional>
#include "transform/acl_ir/acl_helper.h"
#include "ops/structure_op_name.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
constexpr auto kAclOpJitCompile = "acl_op_jit_compile";

bool CustomOpAclKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  converter_ = std::make_shared<transform::AclConverter>();
  MS_EXCEPTION_IF_NULL(converter_);
  MS_EXCEPTION_IF_NULL(primitive_);
  auto op_name = primitive_->GetAttr("reg_op_name");
  if (op_name == nullptr) {
    MS_LOG(EXCEPTION) << "Custom op reg info is error!";
  }
  kernel_name_ = GetValue<std::string>(op_name);
  if (kernel_name_ == "Custom") {
    MS_LOG(EXCEPTION) << "Custom op reg info is error!";
  }
  converter_->runner().SetName(kernel_name_);
  bool jit_config = false;
  if (primitive_->HasAttr(kAclOpJitCompile)) {
    auto op_jit_config = primitive_->GetAttr(kAclOpJitCompile);
    jit_config = GetValue<bool>(op_jit_config);
  }
  if (jit_config) {
    converter_->runner().SetStaticMode();
  } else {
    converter_->runner().SetDynamicMode();
  }
  return true;
}

int CustomOpAclKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(converter_);
  converter_->Reset();
  converter_->runner().ResizeOpInputs(inputs.size());
  converter_->runner().ResizeOpOutputs(outputs.size());

  for (size_t i = 0; i < input_params_.size(); i++) {
    if (i >= inputs.size()) {
      continue;
    }
    auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    auto shape = input->GetShapeVector();
    if (!IsValidShape(shape)) {
      // early stop if any input shape contains -1/-2, which means input shape is dynamic
      MS_LOG(INTERNAL_EXCEPTION) << "For " << kernel_name_ << ", Resize failed because of invalid shape: " << shape;
    }
    input_params_[i].ori_shape = shape;
    input_params_[i].dev_shape = shape;
  }
  for (size_t i = 0; i < output_params_.size(); i++) {
    auto &output = outputs[i];
    MS_EXCEPTION_IF_NULL(output);
    auto shape = output->GetShapeVector();
    if (!IsValidShape(shape)) {
      MS_LOG(EXCEPTION) << "Shape of output " << i << " is invalid, of which value is " << shape
                        << ", node:" << kernel_name_;
    }
    size_t type_size = output_params_[i].type_size;
    size_t tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    tensor_size = std::max(tensor_size, type_size);
    output_size_list_[i] = tensor_size;
    output_params_[i].ori_shape = shape;
    output_params_[i].dev_shape = shape;
  }

  return 0;
}

bool CustomOpAclKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &workspace,
                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }

  MS_EXCEPTION_IF_NULL(converter_);
  MS_EXCEPTION_IF_NULL(primitive_);
  auto input_names_v = primitive_->GetAttr("pure_input_names");
  MS_EXCEPTION_IF_NULL(input_names_v);
  auto input_names = GetValue<std::vector<std::string>>(input_names_v);
  for (size_t i = 0; i < inputs.size(); i++) {
    auto [acl_desc, acl_data] =
      converter_->ConvertTensorToAclDesc(inputs[i], input_params_[i], input_names[i], nullptr, true);
    converter_->runner().SetInput(i, acl_desc, acl_data);
  }

  auto output_names_v = primitive_->GetAttr("output_names");
  MS_EXCEPTION_IF_NULL(output_names_v);
  auto output_names = GetValue<std::vector<std::string>>(output_names_v);
  for (size_t i = 0; i < outputs.size(); i++) {
    auto [acl_desc, acl_data] =
      converter_->ConvertTensorToAclDesc(outputs[i], output_params_[i], output_names[i], nullptr, false);
    converter_->runner().SetOutput(i, acl_desc, acl_data);
  }

  auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
  MS_LOG(DEBUG) << this->DebugString();
  MS_LOG(DEBUG) << converter_->DebugString();
  // release gil before run
  GilReleaseWithCheck release_gil;
  try {
    converter_->Run(stream_ptr);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Kernel launch failed, msg: " << e.what();
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
