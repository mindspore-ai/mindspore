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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_SHARE_SECRETS_KERNEL_H
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_SHARE_SECRETS_KERNEL_H

#include <vector>
#include <string>
#include <memory>
#include "fl/server/common.h"
#include "fl/server/executor.h"
#include "fl/server/kernel/round/round_kernel.h"
#include "fl/server/kernel/round/round_kernel_factory.h"
#include "fl/armour/cipher/cipher_shares.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
class ShareSecretsKernel : public RoundKernel {
 public:
  ShareSecretsKernel() = default;
  ~ShareSecretsKernel() override = default;
  void InitKernel(size_t required_cnt) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;
  bool Reset() override;

 private:
  Executor *executor_;
  size_t iteration_time_window_;
  armour::CipherShares *cipher_share_;
  bool CountForShareSecrets(const std::shared_ptr<FBBuilder> &fbb, const schema::RequestShareSecrets *share_secrets_req,
                            const size_t iter_num);
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_SHARE_SECRETS_KERNEL_H
