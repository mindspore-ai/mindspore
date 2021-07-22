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

#include "src/delegate/npu/op/crop_and_resize_npu.h"
namespace mindspore {
constexpr int BOXES_INDEX = 1;
constexpr int BOX_INDEX = 2;
constexpr int CROP_SIZE_INDEX = 3;
constexpr int CROP_RESIZE_INPUT_SIZE = 4;

int CropAndResizeNPUOp::IsSupport(const schema::Primitive *primitive,
                                  const std::vector<mindspore::MSTensor> &in_tensors,
                                  const std::vector<mindspore::MSTensor> &out_tensors) {
  if (in_tensors.size() < CROP_RESIZE_INPUT_SIZE) {
    MS_LOG(WARNING) << "NPU CropAndResize got inputs size < 4";
    return RET_NOT_SUPPORT;
  }
  auto crop_and_resize_prim = primitive->value_as_CropAndResize();
  if (crop_and_resize_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  // support only 0 linear and 1 nearest
  if (crop_and_resize_prim->method() != schema::ResizeMethod_LINEAR &&
      crop_and_resize_prim->method() != schema::ResizeMethod_NEAREST) {
    MS_LOG(WARNING) << "NPU CropAndResize only support method bilinear 0 and nearest 1, got "
                    << crop_and_resize_prim->method();
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int CropAndResizeNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors) {
  crop_and_resize_ = new (std::nothrow) hiai::op::CropAndResize(name_);
  if (crop_and_resize_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }

  auto crop_and_resize_prim = primitive->value_as_CropAndResize();
  if (crop_and_resize_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  crop_and_resize_->set_attr_extrapolation_value(crop_and_resize_prim->extrapolation_value());
  if (crop_and_resize_prim->method() == schema::ResizeMethod_LINEAR) {
    crop_and_resize_->set_attr_method("bilinear");
  } else if (crop_and_resize_prim->method() == schema::ResizeMethod_NEAREST) {
    crop_and_resize_->set_attr_method("nearest");
  } else {
    MS_LOG(ERROR) << "NPU CropAndResize only support method bilinear and nearest";
    return RET_ERROR;
  }
  return RET_OK;
}

int CropAndResizeNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                     const std::vector<mindspore::MSTensor> &out_tensors,
                                     const std::vector<ge::Operator *> &npu_inputs) {
  crop_and_resize_->set_input_x(*npu_inputs[0]);
  crop_and_resize_->set_input_boxes(*npu_inputs[BOXES_INDEX]);
  crop_and_resize_->set_input_box_index(*npu_inputs[BOX_INDEX]);
  crop_and_resize_->set_input_crop_size(*npu_inputs[CROP_SIZE_INDEX]);
  return RET_OK;
}

ge::Operator *CropAndResizeNPUOp::GetNPUOp() { return this->crop_and_resize_; }

CropAndResizeNPUOp::~CropAndResizeNPUOp() {
  if (crop_and_resize_ != nullptr) {
    delete crop_and_resize_;
    crop_and_resize_ = nullptr;
  }
}
}  // namespace mindspore
