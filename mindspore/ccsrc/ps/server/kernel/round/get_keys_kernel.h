/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_SERVER_KERNEL_GET_KEYS_KERNEL_H
#define MINDSPORE_CCSRC_PS_SERVER_KERNEL_GET_KEYS_KERNEL_H

#include <vector>
#include "ps/server/common.h"
#include "ps/server/kernel/round/round_kernel.h"
#include "ps/server/kernel/round/round_kernel_factory.h"
#include "ps/server/executor.h"
#include "armour/cipher/cipher_keys.h"

namespace mindspore {
namespace ps {
namespace server {
namespace kernel {
class GetKeysKernel : public RoundKernel {
 public:
  GetKeysKernel() = default;
  ~GetKeysKernel() override = default;
  void InitKernel(size_t required_cnt) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  bool Reset() override;

 private:
  Executor *executor_;
  size_t iteration_time_window_;
  armour::CipherKeys *cipher_key_;
};
}  // namespace kernel
}  // namespace server
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_SERVER_KERNEL_GET_KEYS_KERNEL_H
