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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_PULL_WEIGHT_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_PULL_WEIGHT_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "fl/server/common.h"
#include "fl/server/kernel/round/round_kernel.h"
#include "fl/server/kernel/round/round_kernel_factory.h"
#include "fl/server/executor.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
constexpr uint32_t kPrintPullWeightForEveryRetryTime = 500;
class PullWeightKernel : public RoundKernel {
 public:
  PullWeightKernel() : executor_(nullptr), retry_count_(0) {}
  ~PullWeightKernel() override = default;

  void InitKernel(size_t required_cnt) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs);
  bool Reset() override;

 private:
  void PullWeight(const std::shared_ptr<FBBuilder> &fbb, const schema::RequestPullWeight *pull_weight_req);
  void BuildPullWeightRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                          const std::string &reason, size_t iteration,
                          const std::map<std::string, AddressPtr> &feature_maps);

  Executor *executor_;

  // The count of retrying because the aggregation of the weights is not done.
  std::atomic<uint64_t> retry_count_;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_PULL_WEIGHT_KERNEL_H_
