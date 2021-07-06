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

#ifndef MINDSPORE_CCSRC_PS_SERVER_KERNEL_RECONSTRUCT_SECRETS_KERNEL_H_
#define MINDSPORE_CCSRC_PS_SERVER_KERNEL_RECONSTRUCT_SECRETS_KERNEL_H_

#include <vector>
#include <memory>
#include <string>
#include "fl/server/common.h"
#include "fl/server/kernel/round/round_kernel.h"
#include "fl/server/kernel/round/round_kernel_factory.h"
#include "armour/cipher/cipher_reconstruct.h"
#include "fl/server/executor.h"

namespace mindspore {
namespace ps {
namespace server {
namespace kernel {
class ReconstructSecretsKernel : public RoundKernel {
 public:
  ReconstructSecretsKernel() = default;
  ~ReconstructSecretsKernel() override = default;

  void InitKernel(size_t required_cnt) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  bool Reset() override;
  void OnLastCountEvent(const std::shared_ptr<core::MessageHandler> &message) override;

 private:
  std::string name_unmask_;
  Executor *executor_;
  size_t iteration_time_window_;
  armour::CipherReconStruct cipher_reconstruct_;
};
}  // namespace kernel
}  // namespace server
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_SERVER_KERNEL_RECONSTRUCT_SECRETS_KERNEL_H_
