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

#include "frontend/parallel/tensor_layout/reshape_layout_transfer.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/shape_util.h"

namespace mindspore {
namespace parallel {
Status ReshapeLayoutTransfer::CheckValidTransfer() {
  if (!IsSameDeviceArrangement()) {
    return Status::FAILED;
  }
  return Status::SUCCESS;
}

std::shared_ptr<ReshapeLayoutTransfer> ReshapeLayoutTransfer::UnifyDeviceArrangementAndTensorShape() const {
  bool is_unified = IsSameTensorShape();
  std::shared_ptr<ReshapeLayoutTransfer> out_layout_ptr = std::make_shared<ReshapeLayoutTransfer>(*this);
  if (out_layout_ptr == nullptr) {
    return nullptr;
  }
  while (!is_unified) {
    std::shared_ptr<ReshapeLayoutTransfer> temp_layout_ptr = out_layout_ptr->ExtendFromTensorShapeByTo();
    if (temp_layout_ptr == nullptr) {
      out_layout_ptr->SetExpandAble(false);
      return out_layout_ptr;
    }
    out_layout_ptr = temp_layout_ptr->ExtendToTensorShapeByFrom();
    if (out_layout_ptr == nullptr) {
      std::shared_ptr<ReshapeLayoutTransfer> layout_ptr = std::make_shared<ReshapeLayoutTransfer>(*this);
      layout_ptr->SetExpandAble(false);
      return layout_ptr;
    }
    is_unified = out_layout_ptr->IsSameTensorShape();
  }
  return out_layout_ptr;
}

std::shared_ptr<ReshapeLayoutTransfer> ReshapeLayoutTransfer::ExtendFromTensorShapeByTo() const {
  std::shared_ptr<ReshapeLayoutTransfer> out_ptr = std::make_shared<ReshapeLayoutTransfer>(*this);
  bool is_expanded = FromTensorShapeCanBeExpandByTo();
  while (!is_expanded) {
    out_ptr = out_ptr->ExtendFromTensorShapeByExpandedTensorShape();
    if (out_ptr == nullptr) {
      return nullptr;
    }
    is_expanded = out_ptr->FromTensorShapeCanBeExpandByTo();
  }
  return out_ptr;
}

std::shared_ptr<ReshapeLayoutTransfer> ReshapeLayoutTransfer::ExtendToTensorShapeByFrom() const {
  std::shared_ptr<ReshapeLayoutTransfer> out_ptr = std::make_shared<ReshapeLayoutTransfer>(*this);
  bool is_expanded = ToTensorShapeCanBeExpandByFrom();
  while (!is_expanded) {
    out_ptr = out_ptr->ExtendToTensorShapeByExpandedTensorShape();
    if (out_ptr == nullptr) {
      return nullptr;
    }
    is_expanded = out_ptr->ToTensorShapeCanBeExpandByFrom();
  }
  return out_ptr;
}

bool ReshapeLayoutTransfer::FromTensorShapeCanBeExpandByTo() const {
  return from_in_.TensorShapeCanBeExpanded(to_in_.tensor_shape());
}

bool ReshapeLayoutTransfer::ToTensorShapeCanBeExpandByFrom() const {
  return to_in_.TensorShapeCanBeExpanded(from_in_.tensor_shape());
}

std::shared_ptr<ReshapeLayoutTransfer> ReshapeLayoutTransfer::ExtendFromTensorShapeByExpandedTensorShape() const {
  std::shared_ptr<Arrangement> expanded_shape_ptr = ComputeExpandedFromTensorShapeByTo();
  if (expanded_shape_ptr == nullptr) {
    return nullptr;
  }
  return ExpandFromTensorShapeAndExpandToDeviceArrangement(*expanded_shape_ptr);
}

std::shared_ptr<ReshapeLayoutTransfer> ReshapeLayoutTransfer::ExtendToTensorShapeByExpandedTensorShape() const {
  std::shared_ptr<ReshapeLayoutTransfer> exchanged_from_and_to_ptr = ExchangeFromAndTo();
  if (exchanged_from_and_to_ptr == nullptr) {
    return nullptr;
  }
  std::shared_ptr<Arrangement> expanded_shape_ptr = exchanged_from_and_to_ptr->ComputeExpandedFromTensorShapeByTo();
  if (expanded_shape_ptr == nullptr) {
    return nullptr;
  }
  std::shared_ptr<ReshapeLayoutTransfer> exchanged_out =
    exchanged_from_and_to_ptr->ExpandFromTensorShapeAndExpandToDeviceArrangement(*expanded_shape_ptr);
  if (exchanged_out == nullptr) {
    return nullptr;
  }
  return exchanged_out->ExchangeFromAndTo();
}

std::shared_ptr<ReshapeLayoutTransfer> ReshapeLayoutTransfer::ExchangeFromAndTo() const {
  ReshapeLayoutTransfer out;
  Status status = out.Init(to_in_, from_in_);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  return std::make_shared<ReshapeLayoutTransfer>(out);
}

std::shared_ptr<ReshapeLayoutTransfer> ReshapeLayoutTransfer::ExpandFromTensorShapeAndExpandToDeviceArrangement(
  const Arrangement &expand_shape) const {
  std::shared_ptr<TensorLayout> extend_tensor_shape_from_ptr = from_in_.ExpandTensorShape(expand_shape);
  if (extend_tensor_shape_from_ptr == nullptr) {
    return nullptr;
  }
  Arrangement unified_device_arrangement = extend_tensor_shape_from_ptr->device_arrangement();
  std::shared_ptr<TensorLayout> extend_device_arrangement_to_ptr =
    to_in_.ExpandDeviceArrangement(unified_device_arrangement);
  if (extend_device_arrangement_to_ptr == nullptr) {
    return nullptr;
  }
  ReshapeLayoutTransfer out;
  Status status = out.Init(*extend_tensor_shape_from_ptr, *extend_device_arrangement_to_ptr);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  return std::make_shared<ReshapeLayoutTransfer>(out);
}

std::shared_ptr<Arrangement> ReshapeLayoutTransfer::ComputeExpandedFromTensorShapeByTo() const {
  return from_in_.ComputeExpandedTensorShape(to_in_.tensor_shape());
}
}  // namespace parallel
}  // namespace mindspore
