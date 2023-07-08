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
#include "include/backend/op_evaluator.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/kernel.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace evaluator {
namespace {
using KernelTensorPtr = std::shared_ptr<mindspore::kernel::KernelTensor>;
using TensorPtr = std::shared_ptr<tensor::Tensor>;

KernelTensorPtr CreateKernelTensor4Input(const AbstractBasePtr &arg) {
  if (arg->BuildValue() == kValueAny || !arg->isa<mindspore::abstract::AbstractTensor>()) {
    MS_LOG(DEBUG) << "Input arg does not have value or is not a tensor";
    return nullptr;
  }
  // Fetch the dtype from item of tensor.
  auto tensor_abs = dyn_cast<mindspore::abstract::AbstractTensor>(arg);
  MS_EXCEPTION_IF_NULL(tensor_abs->element());
  const auto dtype = tensor_abs->element()->BuildType();
  MS_EXCEPTION_IF_NULL(dtype);
  mindspore::kernel::TensorInfo tensor_info;
  tensor_info.format = mindspore::Format::DEFAULT_FORMAT;
  tensor_info.base_ = tensor_abs;

  auto kernel_tensor = std::make_shared<mindspore::kernel::KernelTensor>();
  kernel_tensor->SetTensorInfo(tensor_info);

  auto tensor_value = dyn_cast<tensor::Tensor>(arg->BuildValue());
  MS_EXCEPTION_IF_NULL(tensor_value);
  MS_LOG(DEBUG) << "Value of input arg is " << tensor_value->ToString() << ", data_c = " << tensor_value->data_c()
                << ", data_size = " << tensor_value->DataSize() << ", size = " << tensor_value->Size();

  mindspore::kernel::AddressPtr addr =
    std::make_shared<mindspore::kernel::Address>(tensor_value->data_c(), tensor_value->Size());
  kernel_tensor->SetData(addr);
  return kernel_tensor;
}

std::pair<KernelTensorPtr, TensorPtr> CreateKernelTensor4Output(const AbstractBasePtr &abs_base) {
  if (!abs_base->isa<mindspore::abstract::AbstractTensor>()) {
    MS_LOG(DEBUG) << "Primtive output is not a tensor";
    return std::pair<KernelTensorPtr, TensorPtr>(nullptr, nullptr);
  }
  // Fetch the dtype from item of tensor.
  auto tensor_abs = dyn_cast<mindspore::abstract::AbstractTensor>(abs_base);
  MS_EXCEPTION_IF_NULL(tensor_abs);
  MS_EXCEPTION_IF_NULL(tensor_abs->element());
  const auto dtype = tensor_abs->element()->BuildType();
  MS_EXCEPTION_IF_NULL(dtype);

  mindspore::kernel::TensorInfo tensor_info;
  tensor_info.format = mindspore::Format::DEFAULT_FORMAT;
  tensor_info.base_ = tensor_abs;

  auto kernel_tensor = std::make_shared<mindspore::kernel::KernelTensor>();
  kernel_tensor->SetTensorInfo(tensor_info);

  auto tensor_shape = dyn_cast<mindspore::abstract::Shape>(tensor_abs->BuildShape());
  MS_LOG(DEBUG) << "output tensor type id = " << dtype->type_id() << ", shape = " << tensor_shape->shape();
  auto out_tensor = std::make_shared<tensor::Tensor>(dtype->type_id(), tensor_shape->shape());
  MS_LOG(DEBUG) << "out_tensor data_c = " << out_tensor->data_c() << ", data_size = " << out_tensor->DataSize()
                << ", size = " << out_tensor->Size() << " info: " << out_tensor->ToString();

  mindspore::kernel::AddressPtr addr =
    std::make_shared<mindspore::kernel::Address>(out_tensor->data_c(), out_tensor->Size());
  kernel_tensor->SetData(addr);

  return std::pair<KernelTensorPtr, TensorPtr>(kernel_tensor, out_tensor);
}

std::shared_ptr<ops::BaseOperator> CreateOperator(const PrimitivePtr &prim) {
  std::string ori_kernel_name =
    prim->HasAttr(kAttrMeOpName) ? GetValue<std::string>(prim->GetAttr(kAttrMeOpName)) : prim->name();
  auto &operator_fns = ops::OperatorRegister::GetInstance().GetOperatorMap();
  auto it = operator_fns.find(ori_kernel_name);
  if (it == operator_fns.end()) {
    MS_LOG(DEBUG) << "Cannot create BaseOperator for " << ori_kernel_name;
    return nullptr;
  }
  auto base_operator = it->second(prim);
  return base_operator;
}

std::pair<bool, TensorPtr> CreateInOutKernelTensor(const AbstractBasePtr &abs_base, const AbstractBasePtrList &args,
                                                   std::vector<KernelTensorPtr> *inputs,
                                                   std::vector<KernelTensorPtr> *outputs) {
  for (auto &arg : args) {
    auto kernel_tensor = CreateKernelTensor4Input(arg);
    if (kernel_tensor == nullptr) {
      return std::pair<bool, TensorPtr>(false, nullptr);
    }
    (void)inputs->emplace_back(kernel_tensor);
  }

  auto [kernel_tensor, tensor] = CreateKernelTensor4Output(abs_base);
  if (kernel_tensor == nullptr) {
    return std::pair<bool, TensorPtr>(false, nullptr);
  }
  (void)outputs->emplace_back(kernel_tensor);

  return std::pair<bool, TensorPtr>(true, tensor);
}
}  // namespace

ValuePtr OpEvaluator::ComputeValue(const PrimitivePtr &prim, const AbstractBasePtr &out,
                                   const AbstractBasePtrList &args) {
  MS_LOG(DEBUG) << "Call cpu backend to infer value from primitive " << prim->name() << " with attribute "
                << GRAPH_FLAG_SIDE_EFFECT_MEM << ": " << GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_MEM);
  auto context = MsContext::GetInstance();
  if (!context->IsSupportDevice(kCPUDevice)) {
    MS_LOG(DEBUG) << "Not support cpu backend or in pynative mode, return AnyValue.";
    return kValueAny;
  }

  if (GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_MEM)) {
    MS_LOG(DEBUG) << "Primitive " << prim->name() << " has side effect, return AnyValue.";
    return kValueAny;
  }

  std::shared_ptr<kernel::NativeCpuKernelMod> cpu_kernel_mod =
    kernel::Factory<kernel::NativeCpuKernelMod>::Instance().Create(prim->name());
  if (cpu_kernel_mod == nullptr) {
    MS_LOG(DEBUG) << "Primitive " << prim->name() << " does not have cpu backend operator, return AnyValue";
    return kValueAny;
  }

  auto op = CreateOperator(prim);
  if (op == nullptr) {
    return kValueAny;
  }

  std::vector<std::shared_ptr<mindspore::kernel::KernelTensor>> inputs;
  std::vector<std::shared_ptr<mindspore::kernel::KernelTensor>> outputs;
  auto [success, out_tensor] = CreateInOutKernelTensor(out, args, &inputs, &outputs);
  if (!success) {
    MS_LOG(DEBUG) << "Can not call cpp infer value, since some inputs have any value";
    return kValueAny;
  }

  if (!cpu_kernel_mod->Init_(op, inputs, outputs)) {
    MS_LOG(EXCEPTION) << "Failed to call cpu kernel module init for primitive " << prim->name();
  }

  if (cpu_kernel_mod->Resize(inputs, outputs) != mindspore::kernel::KRET_OK) {
    MS_LOG(EXCEPTION) << "Failed to call cpu kernel module resize for primitive " << prim->name();
  }

  std::vector<AddressPtr> addr_in, addr_ws, addr_out;
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(addr_in),
                       [](const auto &kernel_tensor) { return kernel_tensor->GetData(); });
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(addr_out),
                       [](const auto &kernel_tensor) { return kernel_tensor->GetData(); });

  if (!cpu_kernel_mod->Launch(addr_in, addr_ws, addr_out, nullptr)) {
    MS_LOG(ERROR) << "Launch cpu kernel module for primitive " << prim->name() << " failed";
    return kValueAny;
  }

  MS_LOG(DEBUG) << "Launch cpu kernel module for primitive " << prim->name() << " success, output is "
                << out_tensor->ToString();
  return out_tensor;
}
}  // namespace evaluator
}  // namespace mindspore
