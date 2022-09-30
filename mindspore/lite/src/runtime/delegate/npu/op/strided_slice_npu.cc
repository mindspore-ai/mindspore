/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/runtime/delegate/npu/op/strided_slice_npu.h"
#include "src/runtime/delegate/npu/npu_converter_utils.h"

namespace mindspore {
int StridedSliceNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                                 const std::vector<mindspore::MSTensor> &out_tensors) {
  // Only onnx StridedSlice has 5 in_tensors, of which the 4th input is axes and the 5th input is strides.
  if (in_tensors.size() == ONNX_INPUT_SIZE) {
    vector<int> axes;
    size_t size = in_tensors[STRIDE_INDEX].Shape()[0];
    axes.resize(size);
    MS_ASSERT(in_tensors[STRIDE_INDEX].Data());
    memcpy(axes.data(), in_tensors[STRIDE_INDEX].Data().get(), sizeof(int) * size);
    for (int i = 0; i < axes.size(); ++i) {
      if (i != axes[i]) {
        MS_LOG(WARNING) << "Does not support setting axis, so the axis must be continuous.";
        return RET_NOT_SUPPORT;
      }
    }
  }
  auto input_x = in_tensors.at(0);
  if (input_x.DataType() != DataType::kNumberTypeFloat32 || input_x.DataType() != DataType::kNumberTypeFloat16) {
    need_cast_ = true;
    MS_LOG(INFO) << "StridedSlice does not support input datatype other than FLOAT. Cast op will be inserted.";
  }
  return RET_OK;
}

int StridedSliceNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors) {
  strided_slice_ = new (std::nothrow) hiai::op::StridedSlice(name_);
  if (strided_slice_ == nullptr) {
    MS_LOG(ERROR) << "New stridedSlice npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  if (need_cast_) {
    in_cast_ = new (std::nothrow) hiai::op::CastT(name_ + "_in_cast");
    out_cast_ = new (std::nothrow) hiai::op::CastT(name_ + "_out_cast");
    if (in_cast_ == nullptr || out_cast_ == nullptr) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }
  auto strided_slice_prim = primitive->value_as_StridedSlice();
  if (strided_slice_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  begins_mask_ = strided_slice_prim->begin_mask();
  ends_mask_ = strided_slice_prim->end_mask();
  ellipsis_mask_ = strided_slice_prim->ellipsis_mask();
  new_axis_mask_ = strided_slice_prim->new_axis_mask();
  shrink_axis_mask_ = strided_slice_prim->shrink_axis_mask();
  return RET_OK;
}

int StridedSliceNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                    const std::vector<mindspore::MSTensor> &out_tensors,
                                    const std::vector<ge::Operator *> &npu_inputs) {
  strided_slice_->set_attr_begin_mask(begins_mask_);
  strided_slice_->set_attr_ellipsis_mask(ellipsis_mask_);
  strided_slice_->set_attr_end_mask(ends_mask_);
  strided_slice_->set_attr_shrink_axis_mask(shrink_axis_mask_);
  strided_slice_->set_attr_new_axis_mask(new_axis_mask_);
  // StridedSliceV2 supports setting axes, but it will cause an endless loop.
  if (need_cast_) {
    in_cast_->set_input_x(*(npu_inputs[0]));
    in_cast_->set_attr_src_dtype(ConverterToNPUDataType(static_cast<DataType>(in_tensors[0].DataType())));
    in_cast_->set_attr_dst_dtype(ge::DT_FLOAT);
    strided_slice_->set_input_x(*in_cast_);
    out_cast_->set_input_x(*strided_slice_);
    out_cast_->set_attr_src_dtype(ge::DT_FLOAT);
    out_cast_->set_attr_dst_dtype(ConverterToNPUDataType(static_cast<DataType>(out_tensors[0].DataType())));
  } else {
    strided_slice_->set_input_x(*npu_inputs[0]);
  }
  strided_slice_->set_input_begin(*npu_inputs[BEGIN_INDEX]);
  strided_slice_->set_input_end(*npu_inputs[END_INDEX]);

  // The strides position of onnx is the 5th, and the others are the 4th.
  if (npu_inputs.size() == ONNX_INPUT_SIZE) {
    strided_slice_->set_input_strides(*npu_inputs[ONNX_STRIDE_INDEX]);
  } else {
    strided_slice_->set_input_strides(*npu_inputs[STRIDE_INDEX]);
  }
  return RET_OK;
}

ge::Operator *StridedSliceNPUOp::GetNPUOp() {
  if (need_cast_) {
    return this->out_cast_;
  } else {
    return this->strided_slice_;
  }
}

int StridedSliceNPUOp::HandleAxisAndConstantInputs(std::vector<mindspore::MSTensor *> *all_tensors) {
  if (inputs_.size() < MIN_INPUT_SIZE) {
    MS_LOG(ERROR) << "StridedSlice in tensors size < " << MIN_INPUT_SIZE;
    return RET_ERROR;
  }
  auto begin_tensor = inputs_.at(BEGIN_INDEX);
  int *begin = reinterpret_cast<int *>(begin_tensor.MutableData());
  MS_ASSERT(begin);
  AssistDataNHWC2NCHW(begin, 1);
  auto end_tensor = inputs_.at(END_INDEX);
  int *end = reinterpret_cast<int *>(end_tensor.MutableData());
  MS_ASSERT(end);
  AssistDataNHWC2NCHW(end, 1);
  auto stride_tensor = inputs_.at(STRIDE_INDEX);
  if (inputs_.size() == ONNX_INPUT_SIZE) {
    stride_tensor = inputs_.at(ONNX_STRIDE_INDEX);
  }
  int *stride = reinterpret_cast<int *>(stride_tensor.MutableData());
  MS_ASSERT(stride);
  AssistDataNHWC2NCHW(stride, 1);
  begins_mask_ = MaskDataNHWC2NCHW(begins_mask_);
  ends_mask_ = MaskDataNHWC2NCHW(ends_mask_);
  ellipsis_mask_ = MaskDataNHWC2NCHW(ellipsis_mask_);
  shrink_axis_mask_ = MaskDataNHWC2NCHW(shrink_axis_mask_);
  new_axis_mask_ = MaskDataNHWC2NCHW(new_axis_mask_);
  return RET_OK;
}

StridedSliceNPUOp::~StridedSliceNPUOp() {
  if (strided_slice_ != nullptr) {
    delete strided_slice_;
    strided_slice_ = nullptr;
  }
  if (in_cast_ != nullptr) {
    delete in_cast_;
    in_cast_ = nullptr;
  }
  if (out_cast_ != nullptr) {
    delete out_cast_;
    out_cast_ = nullptr;
  }
}
}  // namespace mindspore
