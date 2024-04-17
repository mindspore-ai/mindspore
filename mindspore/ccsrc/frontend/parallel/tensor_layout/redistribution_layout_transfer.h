/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_REDISTRIBUTION_LAYOUT_TRANSFER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_REDISTRIBUTION_LAYOUT_TRANSFER_H_

#include <memory>
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/layout_transfer.h"
#include "frontend/parallel/tensor_layout/reshape_layout_transfer.h"

namespace mindspore {
namespace parallel {
class RedistributionLayoutTransfer : public LayoutTransfer {
 public:
  RedistributionLayoutTransfer() = default;
  ~RedistributionLayoutTransfer() override = default;
  std::shared_ptr<ReshapeLayoutTransfer> UnifyDeviceArrangementAndTensorShape() const;

 private:
  Status CheckValidTransfer() override;
  std::shared_ptr<ReshapeLayoutTransfer> UnifyDeviceArrangement() const;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_REDISTRIBUTION_LAYOUT_TRANSFER_H_
