/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_RESHAPE_LAYOUT_TRANSFER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_RESHAPE_LAYOUT_TRANSFER_H_

#include <memory>
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/layout_transfer.h"

namespace mindspore {
namespace parallel {
class ReshapeLayoutTransfer : public LayoutTransfer {
 public:
  ReshapeLayoutTransfer() = default;
  ~ReshapeLayoutTransfer() override = default;
  std::shared_ptr<ReshapeLayoutTransfer> UnifyDeviceArrangementAndTensorShape() const;
  std::shared_ptr<ReshapeLayoutTransfer> ExtendFromTensorShapeByTo() const;
  std::shared_ptr<ReshapeLayoutTransfer> ExtendToTensorShapeByFrom() const;
  std::shared_ptr<ReshapeLayoutTransfer> ExtendFromTensorShapeByExpandedTensorShape() const;
  std::shared_ptr<ReshapeLayoutTransfer> ExtendToTensorShapeByExpandedTensorShape() const;
  std::shared_ptr<ReshapeLayoutTransfer> ExpandFromTensorShapeAndExpandToDeviceArrangement(
    const Arrangement &expand_shape) const;
  std::shared_ptr<ReshapeLayoutTransfer> ExchangeFromAndTo() const;
  bool ExpandAble() const { return is_expand_able_; }
  bool FromTensorShapeCanBeExpandByTo() const;
  bool ToTensorShapeCanBeExpandByFrom() const;
  void SetExpandAble(const bool is_expand_able) { is_expand_able_ = is_expand_able; }

 private:
  Status CheckValidTransfer() override;
  std::shared_ptr<Arrangement> ComputeExpandedFromTensorShapeByTo() const;
  bool is_expand_able_ = true;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_RESHAPE_LAYOUT_TRANSFER_H_
