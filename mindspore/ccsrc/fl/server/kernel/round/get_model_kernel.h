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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_GET_MODEL_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_GET_MODEL_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "fl/server/common.h"
#include "fl/server/executor.h"
#include "fl/server/kernel/round/round_kernel.h"
#include "fl/server/kernel/round/round_kernel_factory.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
constexpr uint32_t kPrintGetModelForEveryRetryTime = 50;
class GetModelKernel : public RoundKernel {
 public:
  GetModelKernel() : executor_(nullptr), iteration_time_window_(0), retry_count_(0) {}
  ~GetModelKernel() override = default;

  void InitKernel(size_t) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs);
  bool Reset() override;

 private:
  void GetModel(const schema::RequestGetModel *get_model_req, const std::shared_ptr<FBBuilder> &fbb);
  void BuildGetModelRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                        const std::string &reason, const size_t iter,
                        const std::map<std::string, AddressPtr> &feature_maps, const std::string &timestamp);

  // The executor is for getting model for getModel request.
  Executor *executor_;

  // The time window of one iteration.
  size_t iteration_time_window_;

  // The count of retrying because the iteration is not finished.
  std::atomic<uint64_t> retry_count_;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_UPDATE_MODEL_KERNEL_H_
