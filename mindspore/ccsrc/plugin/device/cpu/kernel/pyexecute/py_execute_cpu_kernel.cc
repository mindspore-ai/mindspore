/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/pyexecute/py_execute_cpu_kernel.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>

#include "Eigen/Core"
#include "abstract/utils.h"
#include "ir/anf.h"
#include "plugin/device/cpu/hal/device/cpu_common.h"
#include "include/common/fallback.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/utils/python_fallback_running.h"
#include "include/backend/py_execute_utils.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/ccsrc/pipeline/jit/ps/parse/resolve.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace kernel {
bool PyExecuteCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);
  if (primitive_->HasAttr(kAttrPyExecuteNeedUpdateShape)) {
    const auto &value = primitive_->GetAttr(kAttrPyExecuteNeedUpdateShape);
    if (value != nullptr && GetValue<bool>(value)) {
      MS_LOG(INFO) << "pyexecute primitive:" << reinterpret_cast<void *>(primitive_.get()) << " output is not any";
      is_output_any_ = false;
    }
  }
  return true;
}

bool PyExecuteCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                   const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "Launch PyExecute(), inputs.size: " << inputs.size() << ", outputs: " << outputs.size();
  if (Py_IsInitialized() == 0) {
    MS_LOG(ERROR) << "Py_IsInitialized failed.";
    return false;
  }
  MS_LOG(DEBUG) << "The output num is " << outputs.size();
  py::gil_scoped_acquire gil_acquire;
  if (outputs.empty() || outputs[0] == nullptr) {
    MS_LOG(EXCEPTION) << "Invalid output";
  }
  MS_EXCEPTION_IF_NULL(primitive_);
  auto output_value = primitive_->GetAttr(kAttrPyExecuteOutput);
  if (output_value == nullptr) {
    MS_LOG(INFO) << "Prebuilt output result not exists in pyexecute.";
    std::vector<abstract::AbstractBase *> real_inputs;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(real_inputs),
                   [](const auto &real_input) { return real_input; });
    (void)opt::LaunchPy(primitive_, real_inputs);
    output_value = primitive_->GetAttr(kAttrPyExecuteOutput);
    if (output_value == nullptr) {
      MS_LOG(ERROR) << "Rebuilt output result not exists in pyexecute.";
      return false;
    }
  }
  const auto &py_object_output_value = output_value->cast<parse::PyObjectWrapperPtr>();
  MS_EXCEPTION_IF_NULL(py_object_output_value);
  const auto &output = py_object_output_value->obj();
  const auto &output_type = py::str(output.get_type());
  MS_LOG(DEBUG) << "Python *prebuilt* output type: " << output_type << ", output: " << output;
  const auto &py_output = std::make_shared<PyExecuteOutputUserData>();
  constexpr auto kPyExecuteOutIndex = 0;
  py_output->obj = output;
  // Set Python data for kernel node.
  auto out_user_data = output_user_data_.at(kPyExecuteOutIndex);
  out_user_data->set(PyExecuteOutputUserData::key, py_output);
  if (is_output_any_) {
    return true;
  }
  MS_LOG(DEBUG) << "Pyexecute launch for primitive:" << reinterpret_cast<void *>(primitive_.get());
  if (outputs[0]->user_data() == nullptr ||
      outputs[0]->user_data()->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key) == nullptr) {
    MS_LOG(ERROR) << "Invalid output kernel tensor.";
    return false;
  }
  const auto &user_data_obj =
    outputs[0]->user_data()->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
  const auto &obj = user_data_obj->obj;
  try {
    const auto &abstract = pyexecute::GenerateAbstractFromPyObject(obj);
    if (abstract == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to generate abstract for pyexecute";
    }
    MS_LOG(DEBUG) << "Update output shape and type for pyexecute by abstract:" << abstract->ToString();
    if ((!abstract->isa<abstract::AbstractTensor>()) && (!abstract->isa<abstract::AbstractScalar>())) {
      MS_LOG(EXCEPTION) << "Invalid python obj";
    }
    outputs[0]->SetType(abstract->BuildType());
    outputs[0]->SetShape(abstract->BuildShape());
    auto tensor = pyexecute::GetValueByPyObj(obj);
    if (outputs[0]->size() != LongToSize(tensor->data().nbytes())) {
      MS_LOG(EXCEPTION) << "Invalid output size:" << outputs[0]->size()
                        << " and tensor size:" << tensor->data().nbytes();
    }
    const auto &res = memcpy_s(reinterpret_cast<char *>(outputs[0]->device_ptr()), outputs[0]->size(), tensor->data_c(),
                               outputs[0]->size());
    if (res != EOK) {
      MS_LOG(EXCEPTION) << "memcpy failed. res: " << res << ", for tensor:" << tensor->ToString()
                        << " size:" << outputs[0]->size();
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "PyExecute launch failed:" << e.what() << " is any type output:" << is_output_any_
                  << " type:" << (outputs[0]->GetType() == nullptr ? "null" : outputs[0]->GetType()->ToString())
                  << " shape:" << (outputs[0]->GetShape() == nullptr ? "null" : outputs[0]->GetShape()->ToString());
    return false;
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PyExecute, PyExecuteCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
