/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/npu/op/resize_npu.h"
#include <memory>
#include "src/litert/delegate/npu/npu_converter_utils.h"
#include "src/litert/delegate/npu/npu_manager.h"

namespace mindspore::lite {
int ResizeNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors) {
  auto resize_prim = primitive->value_as_Resize();
  if (resize_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  resize_method_ = resize_prim->method();
  if (resize_method_ != schema::ResizeMethod_LINEAR && resize_method_ != schema::ResizeMethod_NEAREST) {
    MS_LOG(WARNING) << "Unsupported resize method type: " << resize_method_;
    return RET_NOT_SUPPORT;
  }
  CHECK_LESS_RETURN(in_tensors.size(), 1);
  CHECK_LESS_RETURN(out_tensors.size(), 1);
  if (in_tensors[0].Shape()[NHWC_H] > out_tensors[0].Shape()[NHWC_H] ||
      in_tensors[0].Shape()[NHWC_W] > out_tensors[0].Shape()[NHWC_W]) {
    MS_LOG(WARNING) << "Npu resize does not support reduction.";
    return RET_NOT_SUPPORT;
  }
  is_support_v2_ = NPUManager::CheckDDKVerGreatEqual("100.500.010.010");
  is_support_scale_ = NPUManager::CheckDDKVerGreatEqual("100.320.012.043");
  return RET_OK;
}

int ResizeNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                      const std::vector<mindspore::MSTensor> &out_tensors) {
  CHECK_LESS_RETURN(in_tensors.at(0).Shape().size(), DIMENSION_4D);
  auto org_height = static_cast<float>(in_tensors.at(0).Shape().at(NHWC_H));
  auto org_width = static_cast<float>(in_tensors.at(0).Shape().at(NHWC_W));
  CHECK_LESS_RETURN(out_tensors.at(0).Shape().size(), DIMENSION_4D);
  auto new_height = static_cast<int>(out_tensors.at(0).Shape().at(NHWC_H));
  auto new_width = static_cast<int>(out_tensors.at(0).Shape().at(NHWC_W));

  ge::TensorPtr size_tensor = std::make_shared<hiai::Tensor>();
  if (is_support_scale_) {
    ge::TensorDesc size_tensor_desc(ge::Shape({NPU_SHAPE_SIZE}), ge::FORMAT_ND, ge::DT_FLOAT);
    size_tensor->SetTensorDesc(size_tensor_desc);
    std::vector<float> data_value = {1, 1, new_height / org_height, new_width / org_width};
    size_tensor->SetData(reinterpret_cast<uint8_t *>(data_value.data()), NPU_SHAPE_SIZE * sizeof(float));
  } else {
    ge::TensorDesc size_tensor_desc(ge::Shape({DIMENSION_2D}), ge::FORMAT_ND, ge::DT_INT32);
    size_tensor->SetTensorDesc(size_tensor_desc);
    std::vector<int> data_value = {new_height, new_width};
    size_tensor->SetData(reinterpret_cast<uint8_t *>(data_value.data()), DIMENSION_2D * sizeof(int));
  }
  out_size_ = new (std::nothrow) hiai::op::Const(name_ + "_size");
  if (out_size_ == nullptr) {
    MS_LOG(ERROR) << "create const NPU op failed for " << name_;
    return RET_ERROR;
  }
  out_size_->set_attr_value(size_tensor);

  auto resize_prim = primitive->value_as_Resize();
  if (resize_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  auto ret = SelectResizeOp(resize_prim);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Select Resize op failed!";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeNPUOp::SelectResizeOp(const mindspore::schema::Resize *prim) {
  if (resize_method_ == schema::ResizeMethod_LINEAR) {
    auto resize_bilinear = new (std::nothrow) hiai::op::ResizeBilinearV2(name_);
    if (resize_bilinear == nullptr) {
      MS_LOG(ERROR) << " resize_ is nullptr.";
      return RET_ERROR;
    }
    resize_bilinear->set_attr_align_corners(prim->coordinate_transform_mode() ==
                                            schema::CoordinateTransformMode_ALIGN_CORNERS);
    resize_bilinear->set_attr_half_pixel_centers(prim->coordinate_transform_mode() ==
                                                 schema::CoordinateTransformMode_HALF_PIXEL);
    resize_bilinear->set_input_size(*out_size_);
    resize_ = resize_bilinear;
  } else if (resize_method_ == schema::ResizeMethod_NEAREST) {
    if (is_support_v2_) {
      auto resize_nearest_v2 = new (std::nothrow) hiai::op::ResizeNearestNeighborV2(name_);
      if (resize_nearest_v2 == nullptr) {
        MS_LOG(ERROR) << " resize_ is nullptr.";
        return RET_ERROR;
      }
      resize_nearest_v2->set_attr_align_corners(prim->coordinate_transform_mode() ==
                                                schema::CoordinateTransformMode_ALIGN_CORNERS);
      resize_nearest_v2->set_input_size(*out_size_);
      resize_ = resize_nearest_v2;
    } else {
      auto resize_nearest = new (std::nothrow) hiai::op::ResizeNearestNeighbor(name_);
      if (resize_nearest == nullptr) {
        MS_LOG(ERROR) << " resize_ is nullptr.";
        return RET_ERROR;
      }
      resize_nearest->set_input_size(*out_size_);
      resize_ = resize_nearest;
    }
  } else {
    MS_LOG(WARNING) << "Unsupported resize method type:" << resize_method_;
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors,
                              const std::vector<ge::Operator *> &npu_inputs) {
  CHECK_LESS_RETURN(npu_inputs.size(), 1);
  if (resize_method_ == schema::ResizeMethod_LINEAR) {
    auto resize_bilinear = reinterpret_cast<hiai::op::ResizeBilinearV2 *>(resize_);
    resize_bilinear->set_input_x(*npu_inputs[0]);
  } else if (resize_method_ == schema::ResizeMethod_NEAREST) {
    if (is_support_v2_) {
      auto resize_nearest_v2 = reinterpret_cast<hiai::op::ResizeNearestNeighborV2 *>(resize_);
      resize_nearest_v2->set_input_x(*npu_inputs[0]);
    } else {
      auto resize_nearest = reinterpret_cast<hiai::op::ResizeNearestNeighbor *>(resize_);
      resize_nearest->set_input_x(*npu_inputs[0]);
    }
  } else {
    MS_LOG(WARNING) << "Unsupported resize method type:" << resize_method_;
    return RET_ERROR;
  }
  return RET_OK;
}

ge::Operator *ResizeNPUOp::GetNPUOp() { return this->resize_; }

ResizeNPUOp::~ResizeNPUOp() {
  if (resize_ != nullptr) {
    delete resize_;
    resize_ = nullptr;
  }
  if (out_size_ != nullptr) {
    delete out_size_;
    out_size_ = nullptr;
  }
}
}  // namespace mindspore::lite
