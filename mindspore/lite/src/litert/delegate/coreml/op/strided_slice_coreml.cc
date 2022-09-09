/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/coreml/op/strided_slice_coreml.h"
#include "src/litert/delegate/delegate_utils.h"
namespace mindspore::lite {
int StridedSliceCoreMLOp::IsSupport() {
  MS_CHECK_TRUE_MSG(in_tensors_.size() < MIN_INPUT_SIZE, RET_NOT_SUPPORT,
                    "StridedSlice input size must be lager than 4");
  // Only onnx StridedSlice has 5 in_tensors, of which the 4th input is axes and the 5th input is strides.
  if (in_tensors_.size() == ONNX_INPUT_SIZE) {
    size_t size = in_tensors_[STRIDE_INDEX].Shape()[0];
    MS_ASSERT(in_tensors_[STRIDE_INDEX].Data());
    auto axes = reinterpret_cast<const int *>(in_tensors_[STRIDE_INDEX].Data().get());
    for (int i = 0; i < size; ++i) {
      if (i != axes[i]) {
        MS_LOG(WARNING) << "Does not support setting axis, so the axis must be continuous.";
        return RET_NOT_SUPPORT;
      }
    }
  }
  for (int i = 1; i < MIN_INPUT_SIZE; ++i) {
    if (!in_tensors_[i].IsConst()) {
      MS_LOG(WARNING) << "Only support static StridedSlice for now, i.e., inputs expect for the first one must be "
                         "constant.";
      return RET_NOT_SUPPORT;
    }
  }
  strided_slice_prim_ = op_primitive_->value_as_StridedSlice();
  if (strided_slice_prim_ == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  if (strided_slice_prim_->ellipsis_mask() != 0 || strided_slice_prim_->new_axis_mask() != 0) {
    MS_LOG(WARNING) << "CoreML StridedSlice dost not support ellipsis_mask and new_axis_mask.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int StridedSliceCoreMLOp::InitParams() {
  auto input_dims = in_tensors_[0].Shape().size();
  begins_idx_ = reinterpret_cast<int *>(in_tensors_.at(BEGIN_INDEX).MutableData());
  ends_idx_ = reinterpret_cast<int *>(in_tensors_.at(END_INDEX).MutableData());
  auto stride_tensor = in_tensors_.at(STRIDE_INDEX);
  if (in_tensors_.size() == ONNX_INPUT_SIZE) {
    stride_tensor = in_tensors_.at(ONNX_STRIDE_INDEX);
  }
  strides_ = reinterpret_cast<int *>(stride_tensor.MutableData());
  begins_mask_ = new bool[input_dims];
  ends_mask_ = new bool[input_dims];
  squeeze_mask_ = new bool[input_dims];
  BinaryMaskData2Bool(strided_slice_prim_->begin_mask(), begins_mask_, input_dims);
  BinaryMaskData2Bool(strided_slice_prim_->end_mask(), ends_mask_, input_dims);
  BinaryMaskData2Bool(strided_slice_prim_->shrink_axis_mask(), squeeze_mask_, input_dims);
  return RET_OK;
}

int StridedSliceCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr && strided_slice_prim_ != nullptr);
  auto strided_slice_param = op_->mutable_slicestatic();
  auto input_dims = in_tensors_[0].Shape().size();
  MS_ASSERT(begins_idx_ != nullptr && ends_idx_ != nullptr && strides_ != nullptr && begins_mask_ != nullptr &&
            ends_mask_ != nullptr && squeeze_mask_ != nullptr);
  for (int i = 0; i < input_dims; ++i) {
    strided_slice_param->add_beginids(begins_idx_[i]);
    strided_slice_param->add_beginmasks(begins_mask_[i]);
    strided_slice_param->add_endids(ends_idx_[i]);
    strided_slice_param->add_endmasks(ends_mask_[i]);
    strided_slice_param->add_strides(strides_[i]);
    strided_slice_param->add_squeezemasks(squeeze_mask_[i]);
  }
  return RET_OK;
}

int StridedSliceCoreMLOp::HandleAxis() {
  // only 4D input case will use this method
  AssistDataNHWC2NCHW<int>(begins_idx_, 1);
  AssistDataNHWC2NCHW<int>(ends_idx_, 1);
  AssistDataNHWC2NCHW<int>(strides_, 1);
  // handle mask
  AssistDataNHWC2NCHW<bool>(begins_mask_, 1);
  AssistDataNHWC2NCHW<bool>(ends_mask_, 1);
  AssistDataNHWC2NCHW<bool>(squeeze_mask_, 1);
  return RET_OK;
}

StridedSliceCoreMLOp::~StridedSliceCoreMLOp() {
  delete begins_mask_;
  begins_mask_ = nullptr;
  delete ends_mask_;
  ends_mask_ = nullptr;
  delete squeeze_mask_;
  squeeze_mask_ = nullptr;
}
}  // namespace mindspore::lite
