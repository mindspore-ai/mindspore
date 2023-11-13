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

#include "frontend/parallel/tensor_layout/redistribution_layout_transfer.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/reshape_layout_transfer.h"
#include "frontend/parallel/tensor_layout/shape_util.h"

namespace mindspore {
namespace parallel {
Status RedistributionLayoutTransfer::CheckValidTransfer() {
  Shape from_shape = from_in_.tensor_shape().array();
  if (std::find(from_shape.begin(), from_shape.end(), -1) != from_shape.end()) {
    is_dynamic_shape_ = true;
    Shape from_map = from_in_.tensor_map().array();
    Shape to_map = to_in_.tensor_map().array();
    bool not_all_repeat = std::any_of(from_map.begin(), from_map.end(), [](int64_t i) { return i != -1; }) ||
                          std::any_of(to_map.begin(), to_map.end(), [](int64_t i) { return i != -1; });
    if (from_in_ != to_in_ && not_all_repeat) {
      MS_LOG(ERROR) << "In dynamic shape scene, the layout between the neighboring node should be equal "
                       "in avoid to insert redistribution operators";
      MS_LOG(ERROR) << "from layout" << from_in_.ToString();
      MS_LOG(ERROR) << "to layout" << to_in_.ToString();
      MS_LOG(ERROR) << "In dynamic shape scene, do not support redistribution";
      return Status::FAILED;
    }
  }
  return Status::SUCCESS;
}

/*
 * unify device arrangement between in_layout and out_layout
 * after this function is called,
 * in_step1_layout.device_arrangement and out_step1_layout.device_arrangement will be the same
 */
std::shared_ptr<ReshapeLayoutTransfer> RedistributionLayoutTransfer::UnifyDeviceArrangement() const {
  Arrangement in_arrangement;
  Arrangement out_arrangement;
  in_arrangement = from_in_.device_arrangement();
  out_arrangement = to_in_.device_arrangement();
  std::shared_ptr<Arrangement> unify_arrangement_ptr = in_arrangement.GetUnifiedShape(out_arrangement);
  if (unify_arrangement_ptr == nullptr) {
    return nullptr;
  }
  std::shared_ptr<TensorLayout> from_out_ptr = from_in_.ExpandDeviceArrangement(*unify_arrangement_ptr);
  if (from_out_ptr == nullptr) {
    return nullptr;
  }
  std::shared_ptr<TensorLayout> to_out_ptr = to_in_.ExpandDeviceArrangement(*unify_arrangement_ptr);
  if (to_out_ptr == nullptr) {
    return nullptr;
  }
  ReshapeLayoutTransfer out;
  Status status = out.Init(*from_out_ptr, *to_out_ptr);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  return std::make_shared<ReshapeLayoutTransfer>(out);
}

/*
 * unify tensor shape between in_step1_layout.tensor_shape and out_step1_layout.tensor_shape
 * after this function is called,
 * in_step2_layout.tensor_shape and out_step2_layout.tensor_shape will be the same
 */
std::shared_ptr<ReshapeLayoutTransfer> RedistributionLayoutTransfer::UnifyDeviceArrangementAndTensorShape() const {
  std::shared_ptr<ReshapeLayoutTransfer> unified_device_arrangement_ptr = UnifyDeviceArrangement();
  if (unified_device_arrangement_ptr == nullptr) {
    ReshapeLayoutTransfer out;
    out.SetExpandAble(false);
    return std::make_shared<ReshapeLayoutTransfer>(out);
  }
  Shape in_expand_shape;
  Status status = ExpandShape(unified_device_arrangement_ptr->from_in().tensor_shape().array(),
                              unified_device_arrangement_ptr->to_in().tensor_shape().array(), &in_expand_shape);
  if (status != Status::SUCCESS) {
    MS_LOG(INFO) << "The shape of from and to cannot transfer by unify";
    unified_device_arrangement_ptr->SetExpandAble(false);
    return unified_device_arrangement_ptr;
  }
  return unified_device_arrangement_ptr->UnifyDeviceArrangementAndTensorShape();
}
}  // namespace parallel
}  // namespace mindspore
