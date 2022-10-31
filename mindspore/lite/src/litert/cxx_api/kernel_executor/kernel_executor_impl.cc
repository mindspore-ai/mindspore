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
#include <string>
#include <unordered_set>
#include <unordered_map>
#include "src/common/ops/ops_utils.h"
#include "src/litert/cxx_api/converters.h"
#include "src/common/prim_util.h"
#include "src/common/ops/populate/populate_register.h"
#include "src/common/primitive_t_utils.h"
#include "schema/inner/model_generated.h"
#include "src/litert/infer_manager.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/cxx_api/kernel_executor/kernel_executor_impl.h"
#include "src/litert/cxx_api/kernel_executor/op_converter.h"
#include "src/litert/cpu_info.h"

namespace mindspore {
namespace {
std::unordered_set<std::string> support_ops = {
  "Abs",       "ReLU",      "Sigmoid", "Add",      "Argmax",  "Argmin",          "AvgPool",
  "BatchNorm", "Ceil",      "Concat",  "Custom",   "Conv2D",  "Conv2DTranspose", "Div",
  "Equal",     "Flatten",   "Gather",  "GatherNd", "MatMul",  "Maximum",         "MaxPool",
  "Minimum",   "Mul",       "Pad",     "PReLU",    "Reshape", "Softmax",         "StridedSlice",
  "TopK",      "Transpose",
};
std::unordered_map<std::string, int> ops_output_num = {
  {"TopK", 2},
};
}  // namespace

KernelExecutorImpl::KernelExecutorImpl() {
  fbb_ = std::make_shared<flatbuffers::FlatBufferBuilder>();
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  lite::CpuInfo cpu_info;
  support_fp16_ = cpu_info.ArmIsSupportFp16();
#endif
}

KernelExecutorImpl::~KernelExecutorImpl() {
  FreeAllResource();
  inputs_.clear();
}

Status KernelExecutorImpl::Build(const std::shared_ptr<ops::BaseOperator> &op, const std::vector<MSTensor> &inputs,
                                 const std::shared_ptr<Context> &ms_context) {
  Status status = BuildInit(op, inputs, ms_context);
  if (status != kSuccess) {
    return status;
  }
  auto op_name = op->name();
  if (support_ops.find(op_name) == support_ops.end()) {
    MS_LOG(ERROR) << "unsupported operator.";
    return kLiteError;
  }
  if (prim_type_ == schema::PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Custom operator need output_num.";
    return kLiteError;
  } else {
    int output_num = ops_output_num.find(op_name) != ops_output_num.end() ? ops_output_num.at(op_name) : 1;
    status = InitTensors(inputs, output_num);
    if (status != kSuccess) {
      return status;
    }
    status = GetCpuKernel(ms_context);
  }

  if (status != kSuccess) {
    MS_LOG(ERROR) << "get cpu kernel error.";
    FreeAllResource();
    return status;
  }
  int ret = kernel_->Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "cpu kernel Prepare error.";
    FreeAllResource();
    return static_cast<StatusCode>(ret);
  }
  return kSuccess;
}

Status KernelExecutorImpl::Build(const std::shared_ptr<ops::Custom> &op, const std::vector<MSTensor> &inputs,
                                 const std::shared_ptr<Context> &ms_context, const int output_num) {
  if (output_num < 1) {
    MS_LOG(ERROR) << "output_num must be greater than 0";
    return kLiteError;
  }
  Status status = BuildInit(op, inputs, ms_context);
  if (status != kSuccess) {
    return status;
  }
  status = InitTensors(inputs, output_num);
  if (status != kSuccess) {
    return status;
  }
  status = GetCustomKernel(ms_context);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "get custom kernel error.";
    FreeAllResource();
    return status;
  }
  int ret = kernel_->Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "custom kernel Prepare error.";
    FreeAllResource();
    return static_cast<StatusCode>(ret);
  }
  return kSuccess;
}

Status KernelExecutorImpl::ReSize(const std::vector<MSTensor> &inputs) {
  if (kernel_ == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return kLiteNullptr;
  }
  if (inputs.size() == 0) {
    MS_LOG(ERROR) << "wrong inputs size.";
    return kLiteError;
  }
  auto status = InitTensors(inputs, 0);
  if (status != kSuccess) {
    return status;
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
    FreeAllResource();
    return static_cast<StatusCode>(ret);
  }
  ret = kernel_->ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "kernel Resize error.";
    FreeAllResource();
    return static_cast<StatusCode>(ret);
  }
  return kSuccess;
}

Status KernelExecutorImpl::Execute(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  if (kernel_ == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return kLiteNullptr;
  }
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "outputs is nullptr.";
    return kLiteNullptr;
  }
  if (inputs.size() != inputs_.size()) {
    MS_LOG(ERROR) << "wrong inputs size.";
    return kLiteError;
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto user_input = inputs[i];
    auto input = inputs_[i];
    if (!TensorIsValid(user_input, input)) {
      MS_LOG(ERROR) << "inputs is invalid.";
      return kLiteError;
    }
    if (user_input.impl() == nullptr) {
      MS_LOG(ERROR) << "Tensor " << user_input.Name() << " is nullptr.";
      return kLiteError;
    }
    auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(user_input.impl());
    inputs_[i] = static_cast<lite::Tensor *>(lite_impl->lite_tensor());
    inputs_[i]->set_category(lite::Category::GRAPH_INPUT);
  }
  kernel_->set_in_tensors(inputs_);
  int ret = kernel_->Execute();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "execute error.";
    return static_cast<StatusCode>(ret);
  }
  auto res = GetOutputs();
  outputs->clear();
  outputs->insert(outputs->end(), res.begin(), res.end());
  return kSuccess;
}

Status KernelExecutorImpl::BuildInit(const std::shared_ptr<ops::BaseOperator> &op, const std::vector<MSTensor> &inputs,
                                     const std::shared_ptr<Context> &ms_context) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "base operator is nullptr.";
    return kLiteNullptr;
  }
  if (inputs.size() == 0) {
    MS_LOG(ERROR) << "wrong inputs size.";
    return kLiteError;
  }
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return kLiteNullptr;
  }
  FreeAllResource();
  auto converter_op = op;
  auto op_creator = lite::OpsConverterRegistry::GetInstance()->GetOpsConverterCreator(op->name());
  if (op_creator != nullptr) {
    converter_op = op_creator(op);
  }
  data_type_ = static_cast<enum TypeId>(inputs[FIRST_INPUT].DataType());
  if (op->name() != "Custom") {
    if (data_type_ != kNumberTypeFloat32 && data_type_ != kNumberTypeFloat16) {
      MS_LOG(ERROR) << "unsupported datatype.";
      return kLiteError;
    } else if (data_type_ == kNumberTypeFloat16 && !support_fp16_) {
      MS_LOG(ERROR) << "inputs dataType is Fp16 but Hw cap NOT support FP16.";
      return kLiteError;
    }
  }

  std::unique_ptr<mindspore::schema::PrimitiveT> prim_t = lite::GetPrimitiveT(converter_op);
  primitive_ = lite::ConvertToPrimitive(prim_t.get(), fbb_.get());
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
  return static_cast<StatusCode>(ret);
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
      get_kernel = lite::KernelRegistry::GetInstance()->GetKernelExec(
        inputs_, outputs_, context_.get(), ms_context.get(), desc, nullptr, &kernel_, primitive_);
    }
  }

  // find kernel only match arch and data_type
  if (get_kernel != RET_OK) {
    kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type_, NHWC, prim_type_, "", ""};
    get_kernel = lite::KernelRegistry::GetInstance()->GetKernelExec(inputs_, outputs_, context_.get(), ms_context.get(),
                                                                    desc, nullptr, &kernel_, primitive_);
  }
  if (get_kernel != RET_OK) {
    MS_LOG(ERROR) << "get custom KernelExec error.";
    return static_cast<StatusCode>(get_kernel);
  }

  int ret = KernelInferShape(inputs_, outputs_, primitive_, context_->GetProviders(), schema_version_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "custom kernel infer shape error. please check inputs size and shape.";
    return static_cast<StatusCode>(ret);
  }
  return kSuccess;
}

Status KernelExecutorImpl::GetCpuKernel(const std::shared_ptr<Context> &ms_context) {
  Status status = GetOpParameter();
  if (status != kSuccess) {
    return status;
  }

  kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type_, NHWC, prim_type_};
  int get_kernel = lite::KernelRegistry::GetInstance()->GetKernelExec(inputs_, outputs_, context_.get(),
                                                                      ms_context.get(), desc, parameter_, &kernel_);
  if (get_kernel != RET_OK) {
    MS_LOG(ERROR) << "get cpu KernelExec error.";
    return static_cast<StatusCode>(get_kernel);
  }

  int ret = KernelInferShape(inputs_, outputs_, parameter_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "cpu kernel infer shape error. please check inputs size and shape.";
    return static_cast<StatusCode>(ret);
  }
  return kSuccess;
}

Status KernelExecutorImpl::InitTensors(const std::vector<MSTensor> &inputs, const int output_num) {
  inputs_.clear();
  for (const auto &tensor : inputs) {
    if (tensor.impl() == nullptr) {
      MS_LOG(ERROR) << "Tensor " << tensor.Name() << " is nullptr.";
      return kLiteNullptr;
    }
    auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(tensor.impl());
    auto lite_tensor = static_cast<lite::Tensor *>(lite_impl->lite_tensor());
    inputs_.emplace_back(lite_tensor);
  }
  for (int i = 0; i < output_num; ++i) {
    lite::Tensor *output_tensor = new (std::nothrow) lite::Tensor();
    if (output_tensor == nullptr) {
      MS_LOG(ERROR) << "Failed to allocate tensor.";
      return kLiteNullptr;
    }
    output_tensor->set_category(lite::Category::VAR);
    outputs_.emplace_back(output_tensor);
  }
  return kSuccess;
}

void KernelExecutorImpl::FreeAllResource() {
  if (kernel_ != nullptr) {
    delete kernel_;
    kernel_ = nullptr;
    // free kernel will free parameter.
    parameter_ = nullptr;
  } else if (parameter_ != nullptr) {
    delete parameter_;
    parameter_ = nullptr;
  }
  for (auto &output : outputs_) {
    if (output != nullptr) {
      delete output;
      output = nullptr;
    }
  }
  outputs_.clear();
}

std::vector<MSTensor> KernelExecutorImpl::GetOutputs() {
  std::vector<MSTensor> empty;
  std::vector<MSTensor> res;
  if (outputs_.empty()) {
    MS_LOG(ERROR) << "The outputs is empty.";
    return empty;
  }
  res.resize(outputs_.size());
  for (size_t i = 0; i < outputs_.size(); i++) {
    auto impl = std::make_shared<LiteTensorImpl>(outputs_[i]);
    if (impl == nullptr || impl->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return empty;
    }
    auto tensor = MSTensor(impl);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return empty;
    }
    res[i] = tensor;
  }
  return res;
}

bool KernelExecutorImpl::TensorIsValid(const MSTensor &ms_tensor, const lite::Tensor *lite_tensor) {
  if (static_cast<enum TypeId>(ms_tensor.DataType()) != lite_tensor->data_type()) {
    MS_LOG(ERROR) << "DataType is invalid.";
    return false;
  }
  if (ms_tensor.format() != lite_tensor->format()) {
    MS_LOG(ERROR) << "Format is invalid.";
    return false;
  }
  auto ms_tensor_shape = ms_tensor.Shape();
  auto lite_tensor_shape = lite_tensor->shape();
  if (ms_tensor_shape.size() != lite_tensor_shape.size()) {
    MS_LOG(ERROR) << "Shape is invalid.";
    return false;
  }
  for (size_t i = 0; i < ms_tensor_shape.size(); i++) {
    if (ms_tensor_shape[i] != lite_tensor_shape[i]) {
      MS_LOG(ERROR) << "Shape is invalid.";
      return false;
    }
  }
  return true;
}
}  // namespace mindspore
