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

#ifndef MINDSPORE_CCSRC_MS_EXTENSION_ASCEND_ATB_COMMON_H_
#define MINDSPORE_CCSRC_MS_EXTENSION_ASCEND_ATB_COMMON_H_

#include <string>
#include <vector>
#include <memory>
#include "acl/acl.h"
#include "atb/atb_infer.h"
#include "atb/infer_op_params.h"
#include "kernel/ascend/custom/pyboost_impl/atb/operation_cache.h"

namespace ms::pynative {
/**
 * @class [API] AtbOpRunner
 * @brief A runner for executing Ascend Transformer Boost (ATB) operations.
 *
 * This class extends the `PyboostRunner` class and provides additional functionality
 * for initializing and running ATB operations on Ascend hardware. It manages input
 * and output tensors, memory allocation, and kernel launches specific to ATB.
 */
class OPS_ASCEND_API AtbOpRunner : public PyboostRunner {
 public:
  using PyboostRunner::PyboostRunner;
  size_t CalcWorkspace() override;
  void LaunchKernel() override;

  /**
   * @brief [API] Initializes the ATB operation with the given parameters.
   *
   * This method retrieves the corresponding ATB operation instance from the
   * `atb::OpParamCache` using the provided parameters and operation name.
   * The operation instance is used to configure and execute the operation.
   *
   * @tparam ParamType The type of the parameter used to configure the ATB operation.
   * @param param The parameters required to initialize the ATB operation.
   *
   * @details
   * - The `param` is passed to the `atb::OpParamCache` singleton, which manages
   *   caching of ATB operation instances.
   * - The operation is retrieved and stored in the private member `op_`.
   */
  template <typename ParamType>
  void Init(const ParamType &param) {
    op_ = atb::AtbContextManager::GetInstance().GetOperation(param, op_name());
  }

 protected:
  void _Run() override;

 private:
  atb::VariantPack variant_pack_;
  uint64_t workspace_size_{0};
  atb::Context *context_{nullptr};
  atb::AtbContextManager::OperationHolder *op_{nullptr};
};

/**
 * @brief [API] Executes an ATB operation using the provided parameters, inputs, and outputs.
 *
 * This function creates an `AtbOpRunner` instance, initializes it with the given parameters,
 * and executes the operation on Ascend hardware.
 *
 * @tparam ParamType The type of the parameter used to configure the ATB operation.
 * @param op_name The name of the ATB operation to execute.
 * @param param The parameters required to initialize the ATB operation.
 * @param inputs A vector of input tensors for the operation.
 * @param outputs A vector of output tensors for the operation.
 *
 * @details
 * - The function first creates a shared pointer to an `AtbOpRunner` instance.
 * - It calls the `Init` method of the runner to initialize the operation.
 * - Finally, it invokes the `Run` method of the runner to execute the operation.
 */
template <typename ParamType>
void RunAtbOp(const std::string &op_name, const ParamType &param, const std::vector<Tensor> &inputs,
              const std::vector<Tensor> &outputs) {
  auto runner = std::make_shared<AtbOpRunner>(op_name);
  runner->Init(param);
  runner->Run(inputs, outputs);
}
}  // namespace ms::pynative
#endif  // MINDSPORE_CCSRC_MS_ EXTENSION_ASCEND_ATB_COMMON_H_
