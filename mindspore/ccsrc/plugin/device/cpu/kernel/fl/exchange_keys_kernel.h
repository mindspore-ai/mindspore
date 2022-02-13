/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_EXCHANGE_KEYS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_EXCHANGE_KEYS_H_

#include <vector>
#include <string>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"
#include "fl/worker/fl_worker.h"
#include "fl/armour/secure_protocol/key_agreement.h"

namespace mindspore {
namespace kernel {
class ExchangeKeysKernelMod : public NativeCpuKernelMod {
 public:
  ExchangeKeysKernelMod() = default;
  ~ExchangeKeysKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;

  void Init(const CNodePtr &kernel_node) override;

  void InitKernel(const CNodePtr &kernel_node) override;

 private:
  bool BuildExchangeKeysReq(const std::shared_ptr<fl::FBBuilder> &fbb);
  std::vector<uint8_t> GetPubicKeyBytes();

  uint32_t rank_id_;
  uint32_t server_num_;
  uint32_t target_server_rank_;
  std::string fl_id_;
  std::shared_ptr<fl::FBBuilder> fbb_;
  armour::PrivateKey *secret_prikey_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_EXCHANGE_KEYS_H_
