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

#ifndef MINDSPORE_LITE_SRC_CXX_API_KERNEL_EXECUTOR_KERNEL_EXECUTOR_IMPL_H
#define MINDSPORE_LITE_SRC_CXX_API_KERNEL_EXECUTOR_KERNEL_EXECUTOR_IMPL_H

#include <vector>
#include <memory>
#include "src/runtime/cxx_api/kernel_executor/kernel_executor.h"
#include "src/runtime/kernel_exec.h"
#include "common/version_manager.h"

namespace mindspore {
class KernelExecutorImpl {
 public:
  KernelExecutorImpl() = default;
  ~KernelExecutorImpl();
  Status Build(const std::shared_ptr<ops::BaseOperator> &op, const std::vector<MSTensor> &inputs,
               const std::vector<MSTensor> &outputs, const std::shared_ptr<Context> &ms_context);
  Status ReSize(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs);
  Status Infer(std::vector<MSTensor> *outputs);
  Status Execute(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs);

 protected:
  Status GetCustomKernel(const std::shared_ptr<Context> &ms_context);
  Status GetCpuKernel(const std::shared_ptr<Context> &ms_context);
  Status GetOpParameter();
  Status InitInOutTensor(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs);
  void FreeInOutTensor();
  bool TensorIsValid(const MSTensor &ms_tensor, const lite::Tensor *lite_tensor);

 private:
  const schema::Primitive *primitive_ = nullptr;
  int prim_type_;
  OpParameter *parameter_ = nullptr;
  lite::InnerContext *context_ = nullptr;
  TypeId data_type_;
  kernel::KernelExec *kernel_ = nullptr;
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  int schema_version_ = lite::SCHEMA_VERSION::SCHEMA_CUR;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_CXX_API_KERNEL_EXECUTOR_KERNEL_EXECUTOR_IMPL_H
