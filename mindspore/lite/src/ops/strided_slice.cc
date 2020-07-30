/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include <vector>
#include "include/errorcode.h"
#include "src/ops/ops.h"
#include "src/runtime/kernel/arm/opclib/op_base.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
namespace {
constexpr int kStridedSliceOutputNum = 1;
constexpr int kStridedSliceInputNum = 1;
}  // namespace

void StridedSlice::ApplyNewAxisMask() {
  for (int i = 0; i < new_axis_mask_.size(); i++) {
    if (new_axis_mask_.at(i)) {
      updated_ndim_ += 1;
      updated_in_shape_.insert(updated_in_shape_.begin() + i, 1);
      updated_begins_.at(i) = 0;
      updated_ends_.at(i) = 1;
      updated_strides_.at(i) = 1;
      updated_begins_.emplace_back(0);
      updated_ends_.emplace_back(updated_in_shape_.at(updated_ndim_ - 1));
      updated_strides_.emplace_back(1);
    }
  }
}

std::vector<int> StridedSlice::ApplyShrinkMask(std::vector<int> out_shape) {
  auto old_out_shape = out_shape;
  out_shape.clear();
  for (int i = 0; i < shrink_axis_mask_.size(); i++) {
    if (shrink_axis_mask_.at(i)) {
      updated_ends_.at(i) = updated_begins_.at(i) + 1;
      updated_strides_.at(i) = 1;
    } else {
      out_shape.emplace_back(old_out_shape.at(i));
    }
  }
  for (int i = shrink_axis_mask_.size(); i < old_out_shape.size(); i++) {
    out_shape.emplace_back(old_out_shape.at(i));
  }
  return out_shape;
}

/*only one bit will be used if multiple bits are true.*/
void StridedSlice::ApplyEllipsisMask() {
  for (int i = 0; i < ellipsis_mask_.size(); i++) {
    if (ellipsis_mask_.at(i)) {
      updated_begins_.at(i) = 0;
      updated_ends_.at(i) = updated_in_shape_.at(i);
      break;
    }
  }
}

void StridedSlice::ApplyBeginMask() {
  for (int i = 0; i < ori_ndim_; i++) {
    updated_begins_.at(i) = 0;
  }
}

void StridedSlice::ApplyEndMask() {
  for (int i = 0; i < ori_ndim_; i++) {
    updated_ends_.at(i) = 0;
  }
}

int StridedSlice::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive != nullptr);
  if (outputs.size() != kStridedSliceOutputNum) {
    MS_LOG(ERROR) << "Invalid output size:" << outputs.size();
    return RET_PARAM_INVALID;
  }
  if (inputs.size() < kStridedSliceInputNum) {
    MS_LOG(ERROR) << "Invalid input size " << inputs.size();
    return RET_PARAM_INVALID;
  }
  auto input = inputs.at(0);
  MS_ASSERT(input != nullptr);
  auto input_shape = input->shape();
  std::vector<int> output_shape;
  auto strided_slice_prim = this->primitive->value_as_StridedSlice();
  updated_ndim_ = static_cast<int>(strided_slice_prim->begin()->size());
  ori_ndim_ = updated_ndim_;
  MS_ASSERT(updated_ndim_ == static_cast<int>(strided_slice_prim->end()->size()));
  MS_ASSERT(updated_ndim_ == static_cast<int>(strided_slice_prim->stride()->size()));
  MS_ASSERT(updated_ndim_ == static_cast<int>(input_shape.size()));

  for (int i = 0; i < updated_ndim_; i++) {
    updated_in_shape_.emplace_back(input_shape.at(i));
    updated_begins_.emplace_back((*(strided_slice_prim->begin()))[i]);
    updated_ends_.emplace_back((*(strided_slice_prim->end()))[i]);
    updated_strides_.emplace_back((*(strided_slice_prim->stride()))[i]);
  }

  // set all mask to original input shape
  begins_mask_.resize(updated_ndim_);
  ends_mask_.resize(updated_ndim_);
  ellipsis_mask_.resize(updated_ndim_);
  new_axis_mask_.resize(updated_ndim_);
  shrink_axis_mask_.resize(updated_ndim_);

  //   convert bit to vector
  for (int i = 0; i < updated_ndim_; i++) {
    begins_mask_.at(i) = static_cast<uint32_t>(strided_slice_prim->beginMask()) & (1 << i);
    ends_mask_.at(i) = static_cast<uint32_t>(strided_slice_prim->endMask()) & (1 << i);
    ellipsis_mask_.at(i) = static_cast<uint32_t>(strided_slice_prim->ellipsisMask()) & (1 << i);
    new_axis_mask_.at(i) = static_cast<uint32_t>(strided_slice_prim->newAxisMask()) & (1 << i);
    shrink_axis_mask_.at(i) = static_cast<uint32_t>(strided_slice_prim->shrinkAxisMask()) & (1 << i);
  }

  ApplyNewAxisMask();
  ApplyNewAxisMask();
  ApplyEndMask();
  ApplyEllipsisMask();

  output_shape.resize(updated_in_shape_.size());
  for (int i = 0; i < updated_in_shape_.size(); i++) {
    if (i < ori_ndim_ && new_axis_mask_.at(i)) {
      output_shape.at(i) = 1;
    } else {
      // begins and ends out of range handling
      if (updated_begins_.at(i) >= updated_in_shape_.at(i) || updated_begins_.at(i) < -updated_in_shape_.at(i) ||
          updated_ends_.at(i) < -updated_in_shape_.at(i) || updated_ends_.at(i) > updated_in_shape_.at(i)) {
        return RET_PARAM_INVALID;
      }
      updated_begins_.at(i) = updated_begins_.at(i) % updated_in_shape_.at(i);
      updated_ends_.at(i) = updated_ends_.at(i) % updated_in_shape_.at(i);

      if ((updated_ends_.at(i) <= updated_begins_.at(i) && updated_strides_.at(i) > 0) ||
          (updated_ends_.at(i) >= updated_begins_.at(i) && updated_strides_.at(i) < 0)) {
        output_shape.at(i) = 0;
      } else {
        output_shape.at(i) = 1 + (updated_ends_.at(i) - updated_begins_.at(i) - 1) / updated_strides_.at(i);
      }
    }
  }

  output_shape = ApplyShrinkMask(output_shape);

  outputs.front()->set_shape(output_shape);
  outputs.front()->set_data_type(input->data_type());
  outputs[0]->SetFormat(input->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
