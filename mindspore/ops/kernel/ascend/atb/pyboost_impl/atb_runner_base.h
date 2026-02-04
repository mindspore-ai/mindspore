/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License")
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
#ifndef MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_ATB_RUNNER_BASE_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_ATB_RUNNER_BASE_H_

#include <vector>
#include <string>
#include <memory>
#include "ir/tensor.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore::kernel::pyboost {
class ATBRunnerBase {
 public:
  ATBRunnerBase() = default;
  ~ATBRunnerBase() = default;

  template <typename ParamType>
  void Init(const std::string &atb_name, ParamType *param) {
    InitProcess(atb_name, param);
  }

  virtual void InitProcess(const std::string &atb_name, void *param_ptr) = 0;

  virtual void GetWorkSpaceInfo(const device::DeviceContext *device_context, uint32_t stream_id,
                                const std::vector<tensor::TensorPtr> &inputs,
                                const std::vector<tensor::TensorPtr> &outputs) = 0;

  virtual void Run(uint32_t stream_id, const device::DeviceContext *device_context) = 0;
};

#define MS_ATB_RUNNER_FACTORY_REG(NAME, DERIVE) MS_KERNEL_FACTORY_REG(ATBRunnerBase, NAME, DERIVE)
}  // namespace mindspore::kernel::pyboost
#endif  // MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_ATB_RUNNER_BASE_H_
