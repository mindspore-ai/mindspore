/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include "src/common/ops/ops_utils.h"
#include "src/runtime/cxx_api/converters.h"
#include "src/common/prim_util.h"
#include "src/common/ops/populate/populate_register.h"
#include "src/common/primitive_t_utils.h"
#include "schema/inner/model_generated.h"
#include "src/runtime/infer_manager.h"
#include "src/runtime/kernel_registry.h"
#include "src/runtime/cxx_api/kernel_executor/kernel_executor_impl.h"

namespace mindspore {
constexpr size_t INITIAL_SIZE = 1024;

KernelExecutorImpl::~KernelExecutorImpl() {
  if (context_ != nullptr) {
    delete context_;
    context_ = nullptr;
  }

  if (kernel_ != nullptr) {
    delete kernel_;
    kernel_ = nullptr;
  }
  FreeInOutTensor();
}

Status KernelExecutorImpl::Build(const std::shared_ptr<ops::BaseOperator> &op, const std::vector<MSTensor> &inputs,
                                 const std::vector<MSTensor> &outputs, const std::shared_ptr<Context> &ms_context) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "base operator is nullptr.";
    return kLiteNullptr;
  }
  if (inputs.size() == 0) {
    MS_LOG(ERROR) << "wrong inputs size.";
    return kLiteError;
  }
  if (outputs.size() == 0) {
    MS_LOG(ERROR) << "wrong outputs size.";
    return kLiteError;
  }
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return kLiteNullptr;
  }
  data_type_ = static_cast<enum TypeId>(inputs[FIRST_INPUT].DataType());
  std::unique_ptr<mindspore::schema::PrimitiveT> prim_t = lite::GetPrimitiveT(op);
  flatbuffers::FlatBufferBuilder fbb(INITIAL_SIZE);
  primitive_ = lite::ConvertToPrimitive(prim_t.get(), &fbb);
  fbb.Clear();
  if (primitive_ == nullptr) {
    MS_LOG(ERROR) << "convert to primitive nullptr.";
    return kLiteNullptr;
  }
  prim_type_ = lite::GetPrimitiveType(primitive_, schema_version_);

  context_ = ContextUtils::Convert(ms_context.get());
  if (context_ == nullptr) {
    MS_LOG(ERROR) << "failed to convert Context to LiteContext.";
    return kLiteNullptr;
  }
  int ret = context_->Init();
  if (ret != RET_OK) {
    return static_cast<StatusCode>(ret);
  }

  Status status = InitInOutTensor(inputs, outputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "InitInOutTensor error.";
    return status;
  }

  if (prim_type_ == schema::PrimitiveType_Custom) {
    status = GetCustomKernel(ms_context);
  } else {
    status = GetCpuKernel(ms_context);
  }

  if (status != kSuccess) {
    MS_LOG(ERROR) << "get kernel error.";
    return status;
  }
  ret = kernel_->Prepare();
  return static_cast<StatusCode>(ret);
}

Status KernelExecutorImpl::ReSize(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs) {
  if (inputs.size() == 0) {
    MS_LOG(ERROR) << "wrong inputs size.";
    return kLiteError;
  }
  if (outputs.size() == 0) {
    MS_LOG(ERROR) << "wrong outputs size.";
    return kLiteError;
  }
  Status status = InitInOutTensor(inputs, outputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "InitInOutTensor error.";
    return status;
  }
  if (kernel_ == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return kLiteNullptr;
  }
  kernel_->set_in_tensors(inputs_);
  kernel_->set_out_tensors(outputs_);
  int ret;
  if (kernel_->type() == schema::PrimitiveType_Custom) {
    ret = KernelInferShape(inputs_, outputs_, primitive_, context_->GetProviders(), schema_version_);
  } else {
    ret = KernelInferShape(inputs_, outputs_, parameter_);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "do infer shape error.";
    return static_cast<StatusCode>(ret);
  }
  ret = kernel_->ReSize();
  return static_cast<StatusCode>(ret);
}
Status KernelExecutorImpl::Infer(std::vector<MSTensor> *outputs) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "outputs is nullptr.";
    return kLiteNullptr;
  }
  if (outputs->size() != outputs_.size()) {
    MS_LOG(ERROR) << "wrong outputs size.";
    return kLiteError;
  }
  for (size_t i = 0; i < outputs->size(); ++i) {
    auto user_output = outputs->at(i);
    auto output = outputs_[i];
    user_output.SetFormat(output->format());
    auto output_shape = output->shape();
    std::vector<int64_t> shape;
    std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(shape),
                   [](auto s) { return static_cast<int64_t>(s); });
    user_output.SetShape(shape);
  }
  return kSuccess;
}

Status KernelExecutorImpl::Execute(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs) {
  if (inputs.size() != inputs_.size()) {
    MS_LOG(ERROR) << "wrong inputs size.";
    return kLiteError;
  }
  if (outputs.size() != outputs_.size()) {
    MS_LOG(ERROR) << "wrong outputs size.";
    return kLiteError;
  }
  if (kernel_ == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return kLiteNullptr;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto user_input = inputs[i];
    auto input = inputs_[i];
    if (!TensorIsValid(user_input, input)) {
      MS_LOG(ERROR) << "inputs is invalid.";
      return kLiteError;
    }
    input->set_data(user_input.MutableData());
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto user_output = outputs[i];
    auto output = outputs_[i];
    if (!TensorIsValid(user_output, output)) {
      MS_LOG(ERROR) << "outputs is invalid.";
      return kLiteError;
    }
    output->set_data(user_output.MutableData());
  }
  int ret = kernel_->Execute();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "execute error.";
    return static_cast<StatusCode>(ret);
  }

  return kSuccess;
}

Status KernelExecutorImpl::GetOpParameter() {
  auto parame_gen = lite::PopulateRegistry::GetInstance()->GetParameterCreator(prim_type_, schema_version_);
  if (parame_gen == nullptr) {
    MS_LOG(ERROR) << "parameter generator is nullptr.";
    return kLiteNullptr;
  }
  parameter_ = parame_gen(primitive_);
  if (parameter_ == nullptr) {
    MS_LOG(ERROR) << "PopulateParameter return nullptr, type: "
                  << lite::GetPrimitiveTypeName(primitive_, schema_version_);
    return kLiteNullptr;
  }
  parameter_->thread_num_ = context_->thread_num_;
  return kSuccess;
}

Status KernelExecutorImpl::GetCustomKernel(const std::shared_ptr<Context> &ms_context) {
  int get_kernel = lite::RET_ERROR;

  // find kernel match arch, data_type, kernel_arch and provider
  for (auto &&device : context_->device_list_) {
    if (!device.provider_.empty() && !device.provider_device_.empty()) {
      kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type_,      NHWC, prim_type_,
                             device.provider_device_,   device.provider_};
      get_kernel = lite::KernelRegistry::GetInstance()->GetKernelExec(inputs_, outputs_, context_, ms_context.get(),
                                                                      desc, nullptr, &kernel_, primitive_);
    }
  }

  // find kernel only match arch and data_type
  if (get_kernel != RET_OK) {
    kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type_, NHWC, prim_type_, "", ""};
    get_kernel = lite::KernelRegistry::GetInstance()->GetKernelExec(inputs_, outputs_, context_, ms_context.get(), desc,
                                                                    nullptr, &kernel_, primitive_);
  }

  // if found kernel, do infershape
  if (get_kernel == RET_OK) {
    int ret = KernelInferShape(inputs_, outputs_, primitive_, context_->GetProviders(), schema_version_);
    return static_cast<StatusCode>(ret);
  }

  return static_cast<StatusCode>(get_kernel);
}

Status KernelExecutorImpl::GetCpuKernel(const std::shared_ptr<Context> &ms_context) {
  Status status = GetOpParameter();
  if (status != kSuccess) {
    return status;
  }

  kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type_, NHWC, prim_type_};
  int get_kernel = lite::KernelRegistry::GetInstance()->GetKernelExec(inputs_, outputs_, context_, ms_context.get(),
                                                                      desc, parameter_, &kernel_);
  if (get_kernel == RET_OK) {
    int ret = KernelInferShape(inputs_, outputs_, parameter_);
    return static_cast<StatusCode>(ret);
  }

  return static_cast<StatusCode>(get_kernel);
}

void KernelExecutorImpl::FreeInOutTensor() {
  for (auto &input : inputs_) {
    if (input != nullptr) {
      delete input;
      input = nullptr;
    }
  }
  inputs_.clear();
  for (auto &output : outputs_) {
    if (output != nullptr) {
      delete output;
      output = nullptr;
    }
  }
  outputs_.clear();
}

Status KernelExecutorImpl::InitInOutTensor(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs) {
  FreeInOutTensor();
  for (auto input : inputs) {
    auto input_shape = input.Shape();
    std::vector<int> shape;
    std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(shape),
                   [](auto s) { return static_cast<int>(s); });
    lite::Tensor *input_tensor = new (std::nothrow)
      lite::Tensor(static_cast<enum TypeId>(input.DataType()), shape, input.format(), lite::Category::GRAPH_INPUT);
    if (input_tensor == nullptr) {
      delete input_tensor;
      return kLiteNullptr;
    }
    input_tensor->set_data(input.MutableData());
    inputs_.emplace_back(input_tensor);
  }

  for (auto output : outputs) {
    auto output_shape = output.Shape();
    std::vector<int> shape;
    std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(shape),
                   [](auto s) { return static_cast<int>(s); });
    lite::Tensor *output_tensor =
      new (std::nothrow) lite::Tensor(static_cast<enum TypeId>(output.DataType()), shape, output.format());
    if (output_tensor == nullptr) {
      delete output_tensor;
      return kLiteNullptr;
    }
    outputs_.emplace_back(output_tensor);
  }
  return kSuccess;
}

bool KernelExecutorImpl::TensorIsValid(const MSTensor &ms_tensor, const lite::Tensor *lite_tensor) {
  if (static_cast<enum TypeId>(ms_tensor.DataType()) != lite_tensor->data_type()) {
    return false;
  }
  if (ms_tensor.format() != lite_tensor->format()) {
    return false;
  }
  auto ms_tensor_shape = ms_tensor.Shape();
  auto lite_tensor_shape = lite_tensor->shape();
  if (ms_tensor_shape.size() != lite_tensor_shape.size()) {
    return false;
  }
  for (size_t i = 0; i < ms_tensor_shape.size(); i++) {
    if (ms_tensor_shape[i] != lite_tensor_shape[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace mindspore
