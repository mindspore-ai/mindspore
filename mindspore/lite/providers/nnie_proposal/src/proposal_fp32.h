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

#ifndef MINDSPORE_LITE_PROVIDERS_NNIE_PROPOSAL_SRC_PROPOSAL_FP32_H_
#define MINDSPORE_LITE_PROVIDERS_NNIE_PROPOSAL_SRC_PROPOSAL_FP32_H_

#include <vector>
#include "schema/model_generated.h"
#include "include/api/kernel.h"
#include "src/proposal.h"

using mindspore::kernel::Kernel;
namespace mindspore {
namespace proposal {
class ProposalCPUKernel : public Kernel {
 public:
  ProposalCPUKernel(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs,
                    const mindspore::schema::Primitive *primitive, const mindspore::Context *ctx, int id,
                    int image_height, int image_width)
      : Kernel(inputs, outputs, primitive, ctx), id_(id), image_height_(image_height), image_weight_(image_width) {}

  ~ProposalCPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Execute() override;

 private:
  proposal::ProposalParam proposal_param_ = {0};
  int64_t id_;
  int64_t image_height_;
  int64_t image_weight_;
};
}  // namespace proposal
}  // namespace mindspore

#endif  // MINDSPORE_LITE_PROVIDERS_NNIE_PROPOSAL_SRC_PROPOSAL_FP32_H_
