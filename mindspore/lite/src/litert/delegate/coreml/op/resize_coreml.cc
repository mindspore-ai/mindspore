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

#include "src/litert/delegate/coreml/op/resize_coreml.h"
namespace mindspore::lite {
int ResizeCoreMLOp::IsSupport() {
  resize_prim_ = op_primitive_->value_as_Resize();
  if (resize_prim_ == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  auto resize_method = resize_prim_->method();
  if (resize_method != schema::ResizeMethod_LINEAR && resize_method != schema::ResizeMethod_NEAREST) {
    MS_LOG(WARNING) << "Unsupported resize method type: " << resize_method;
    return RET_NOT_SUPPORT;
  }
  if (resize_method != schema::ResizeMethod_LINEAR ||
      resize_prim_->coordinate_transform_mode() != schema::CoordinateTransformMode_ALIGN_CORNERS) {
    use_upsample_ = true;
    if (in_tensors_.size() != kInputSize1 || !in_tensors_[1].IsConst() || in_tensors_[1].ElementNum() != C2NUM) {
      MS_LOG(WARNING) << "The second input must be a constant with two scale values of height and width when using "
                         "CoreML upsample layer for op: "
                      << name_;
      return RET_NOT_SUPPORT;
    }
  }
  return RET_OK;
}

int ResizeCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr);
  if (use_upsample_) {
    auto resize_param = op_->mutable_upsample();
    MS_CHECK_GE(in_tensors_.size(), kInputSize1, RET_NOT_SUPPORT);
    auto scale_tensor = in_tensors_.at(1);
    auto scale_data = scale_tensor.Data().get();
    if (scale_tensor.DataType() == DataType::kNumberTypeInt32) {
      resize_param->add_scalingfactor(static_cast<const int *>(scale_data)[0]);
      resize_param->add_scalingfactor(static_cast<const int *>(scale_data)[1]);
    } else if (scale_tensor.DataType() == DataType::kNumberTypeFloat32) {
      resize_param->add_fractionalscalingfactor(static_cast<const float *>(scale_data)[0]);
      resize_param->add_fractionalscalingfactor(static_cast<const float *>(scale_data)[1]);
    } else {
      MS_LOG(ERROR) << "Unsupported Resize scale data type: " << static_cast<int>(scale_tensor.DataType());
      return RET_ERROR;
    }
    if (resize_prim_->method() == schema::ResizeMethod_LINEAR) {
      resize_param->set_mode(CoreML::Specification::UpsampleLayerParams_InterpolationMode_BILINEAR);
      if (resize_prim_->coordinate_transform_mode() == schema::CoordinateTransformMode_ALIGN_CORNERS) {
        resize_param->set_linearupsamplemode(
          CoreML::Specification::UpsampleLayerParams_LinearUpsampleMode_ALIGN_CORNERS_TRUE);
      } else {
        resize_param->set_linearupsamplemode(
          CoreML::Specification::UpsampleLayerParams_LinearUpsampleMode_ALIGN_CORNERS_FALSE);
      }
    } else if (resize_prim_->method() == schema::ResizeMethod_NEAREST) {
      resize_param->set_mode(CoreML::Specification::UpsampleLayerParams_InterpolationMode_NN);
    }
    return RET_OK;
  }
  // Using resize_bilinear op. The op executed with NCHW format.
  auto out_height = static_cast<int>(out_tensors_.at(0).Shape().at(kNCHW_H));
  auto out_width = static_cast<int>(out_tensors_.at(0).Shape().at(kNCHW_W));
  auto resize_param = op_->mutable_resizebilinear();
  resize_param->add_targetsize(out_height);
  resize_param->add_targetsize(out_width);
  resize_param->mutable_mode()->set_samplingmethod(CoreML::Specification::SamplingMode::STRICT_ALIGN_ENDPOINTS_MODE);
  return RET_OK;
}
}  // namespace mindspore::lite
