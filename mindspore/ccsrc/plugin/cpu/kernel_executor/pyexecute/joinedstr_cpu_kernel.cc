/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "plugin/cpu/kernel_executor/pyexecute/joinedstr_cpu_kernel.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>

#include "Eigen/Core"
#include "ir/anf.h"
#include "primitive/framework_op_name.h"
#include "utils/log_adapter.h"
#include "include/utils/convert_utils_py.h"
#include "frontend/jit/ps/parse/resolve.h"
#include "include/utils/pipeline/fallback.h"
#include "include/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
namespace {
ValuePtr GetValueByAbstract(const abstract::AbstractBase *abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<kernel::KernelTensor>()) {
    MS_LOG(EXCEPTION) << "Invalid kernel tensor:" << abstract->ToString();
  }
  const auto &kernel_tensor = dynamic_cast<const kernel::KernelTensor *>(abstract);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (kernel_tensor->user_data() != nullptr) {
    auto user_data =
      kernel_tensor->user_data()->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
    MS_EXCEPTION_IF_NULL(user_data);
    return std::make_shared<parse::PyObjectWrapper>(user_data->obj, "graph python obj");
  }

  if (kernel_tensor->GetValueTrack() != nullptr && !kernel_tensor->GetValueTrack()->isa<ValueAny>()) {
    return kernel_tensor->GetValueTrack();
  } else if (IsShapeEmpty(kernel_tensor->GetShapeVector())) {
    auto type_id =
      (kernel_tensor->dtype_id() == TypeId::kTypeUnknown ? TypeId::kNumberTypeInt64 : kernel_tensor->dtype_id());
    return std::make_shared<tensor::Tensor>(type_id, kernel_tensor->GetShapeVector());
  }

  MS_LOG(DEBUG) << "Type:" << kernel_tensor->dtype_id() << " shape:" << kernel_tensor->GetShapeVector()
                << " size:" << kernel_tensor->size();
  auto real_value = kernel_tensor->GetValue();
  MS_EXCEPTION_IF_NULL(real_value);
  if (!real_value->isa<KernelTensorValue>()) {
    MS_LOG(EXCEPTION) << "Invalid kernel tensor value:" << real_value->ToString();
  }

  auto kernel_tensor_value = real_value->cast<KernelTensorValuePtr>();
  MS_EXCEPTION_IF_NULL(kernel_tensor_value);
  if (kernel_tensor->GetType() != nullptr && kernel_tensor->GetType()->isa<Number>()) {
    return common::AnfAlgo::ValueToScalar(kernel_tensor_value, kernel_tensor->GetType()->type_id());
  }

  tensor::TensorPtr tensor =
    std::make_shared<tensor::Tensor>(kernel_tensor->dtype_id(), kernel_tensor->GetShapeVector());
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->DataNBytes() != kernel_tensor_value->GetDataSize()) {
    MS_LOG(EXCEPTION) << "Invalid host tensor size:" << tensor->DataNBytes()
                      << " and kernel tensor size:" << kernel_tensor_value->GetDataSize() << " for pyexecute.";
  }
  auto data_ptr = tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  const auto &res = memcpy_s(data_ptr, kernel_tensor_value->GetDataSize(), kernel_tensor_value->GetDataPtr(),
                             kernel_tensor_value->GetDataSize());
  if (res != EOK) {
    MS_LOG(EXCEPTION) << "memcpy failed. res: " << res << ", for tensor:" << tensor->ToString()
                      << " size:" << kernel_tensor_value->GetDataSize();
  }
  return tensor;
}
}  // namespace

bool JoinedStrCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "The input size is " + std::to_string(inputs.size());
  MS_EXCEPTION_IF_NULL(primitive_);
  return true;
}

std::string ConvertAbsToStr(KernelTensor *input) {
  auto py_tensor = ValueToPyData(GetValueByAbstract(input));
  MS_EXCEPTION_IF_NULL(py_tensor);
  return py::str(py_tensor).cast<std::string>();
}

bool JoinedStrCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                   const std::vector<KernelTensor *> &outputs) {
  py::gil_scoped_acquire gil_acquire;
  std::string exception_msg;
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    exception_msg += ConvertAbsToStr(input);
  }
  AbstractBase *output = outputs[0];
  output->set_user_data<string>("str_exception_result", std::make_shared<string>(exception_msg));
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, JoinedStr, JoinedStrCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
