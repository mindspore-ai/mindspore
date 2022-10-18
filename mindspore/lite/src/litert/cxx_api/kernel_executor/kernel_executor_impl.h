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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_CXX_API_KERNEL_EXECUTOR_KERNEL_EXECUTOR_IMPL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_CXX_API_KERNEL_EXECUTOR_KERNEL_EXECUTOR_IMPL_H_

#include <vector>
#include <memory>
#include "src/litert/cxx_api/kernel_executor/kernel_executor.h"
#include "src/litert/kernel_exec.h"
#include "common/version_manager.h"

namespace mindspore {
class KernelExecutorImpl {
 public:
  KernelExecutorImpl() = default;
  ~KernelExecutorImpl();
  Status Build(const std::shared_ptr<ops::BaseOperator> &op, const std::vector<MSTensor> &inputs,
               const std::shared_ptr<Context> &ms_context);
  Status Build(const std::shared_ptr<ops::Custom> &op, const std::vector<MSTensor> &inputs,
               const std::shared_ptr<Context> &ms_context, const int output_num);
  Status ReSize(const std::vector<MSTensor> &inputs);
  Status Execute(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);

 protected:
  Status BuildInit(const std::shared_ptr<ops::BaseOperator> &op, const std::vector<MSTensor> &inputs,
                   const std::shared_ptr<Context> &ms_context);
  Status GetCustomKernel(const std::shared_ptr<Context> &ms_context);
  Status GetCpuKernel(const std::shared_ptr<Context> &ms_context);
  Status GetOpParameter();
  void InitTensors(const std::vector<MSTensor> &inputs, const int output_num);
  void FreeAllResource();
  std::vector<MSTensor> GetOutputs();
  bool TensorIsValid(const MSTensor &ms_tensor, const lite::Tensor *lite_tensor);
  void Int8TensorAddQuantParam(lite::Tensor *lite_tensor);

 private:
  const schema::Primitive *primitive_ = nullptr;
  int prim_type_;
  OpParameter *parameter_ = nullptr;
  std::shared_ptr<lite::InnerContext> context_ = nullptr;
  TypeId data_type_;
  kernel::KernelExec *kernel_ = nullptr;
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  int schema_version_ = lite::SCHEMA_VERSION::SCHEMA_CUR;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_CXX_API_KERNEL_EXECUTOR_KERNEL_EXECUTOR_IMPL_H_
