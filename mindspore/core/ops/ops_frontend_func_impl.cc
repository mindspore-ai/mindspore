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

#include "ops/ops_frontend_func_impl.h"
#include "utils/log_adapter.h"

namespace mindspore::ops {
OpsFrontendFuncImplMap *GetOpsFrontendFuncImplMapPtr() {
  static OpsFrontendFuncImplMap ops_frontend_func_impl_map;
  return &ops_frontend_func_impl_map;
}

OpFrontendFuncImplPtr GetOpFrontendFuncImplPtr(const std::string &name) {
  auto iter = GetOpsFrontendFuncImplMapPtr()->find(name);
  if (iter == GetOpsFrontendFuncImplMapPtr()->end()) {
    return nullptr;
  }

  return iter->second.get_func_impl();
}

RegFrontendFuncImplHelper::RegFrontendFuncImplHelper(const std::string &name, const OpFrontendFuncImplPtr &func_impl) {
  const FrontendFuncImplHolder holder{func_impl};
  (void)GetOpsFrontendFuncImplMapPtr()->emplace(name, holder);
}

InferValueCallback &InferValueCallback::GetInstance() {
  static InferValueCallback instance{};
  return instance;
}

void InferValueCallback::RegImpl(const std::string &impl_type, const InferValueFunc &func) {
  if (impl_type == "python_impl") {
    if (python_impl_) {
      MS_LOG(ERROR) << "InferValueImpl for python_impl is already registered!";
    }
    python_impl_ = func;
  } else if (impl_type == "cpu_kernel_impl") {
    if (kernel_impl_) {
      MS_LOG(ERROR) << "InferValueImpl for cpu_kernel_impl is already registered!";
    }
    kernel_impl_ = func;
  } else {
    MS_LOG(ERROR) << "Unsupported InferValue implement type " << impl_type << "!";
  }
}

ValuePtr InferValueCallback::CallPyInferValue(const std::string &op_name, const AbstractBasePtrList &input_args) {
  if (python_impl_) {
    return python_impl_(op_name, input_args);
  }
  return nullptr;
}
ValuePtr InferValueCallback::CallKernelInferValue(const std::string &op_name, const AbstractBasePtrList &input_args) {
  if (kernel_impl_) {
    return kernel_impl_(op_name, input_args);
  }
  return nullptr;
}

InferValueImplRegister::InferValueImplRegister(const std::string &impl_type, const InferValueFunc &fn) {
  InferValueCallback::GetInstance().RegImpl(impl_type, fn);
}
}  //  namespace mindspore::ops
