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

#include "src/ops/resize.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Resize::GetFormat() const { return this->primitive_->value.AsResize()->format; }
int Resize::GetMethod() const { return this->primitive_->value.AsResize()->method; }
int64_t Resize::GetNewHeight() const { return this->primitive_->value.AsResize()->newHeight; }
int64_t Resize::GetNewWidth() const { return this->primitive_->value.AsResize()->newWidth; }
bool Resize::GetPreserveAspectRatio() const { return this->primitive_->value.AsResize()->preserveAspectRatio; }
int Resize::GetCoordinateTransformMode() const { return this->primitive_->value.AsResize()->coordinateTransformMode; }

void Resize::SetFormat(int format) { this->primitive_->value.AsResize()->format = (schema::Format)format; }
void Resize::SetMethod(int method) { this->primitive_->value.AsResize()->method = (schema::ResizeMethod)method; }
void Resize::SetNewHeight(int64_t new_height) { this->primitive_->value.AsResize()->newHeight = new_height; }
void Resize::SetNewWidth(int64_t new_width) { this->primitive_->value.AsResize()->newWidth = new_width; }
void Resize::SetCoordinateTransformMode(int coordinate_transform_mode) {
  this->primitive_->value.AsResize()->coordinateTransformMode =
    static_cast<schema::CoordinateTransformMode>(coordinate_transform_mode);
}
void Resize::SetPreserveAspectRatio(bool preserve_aspect_ratio) {
  this->primitive_->value.AsResize()->preserveAspectRatio = preserve_aspect_ratio;
}

int Resize::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Resize;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Resize) {
    MS_LOG(ERROR) << "primitive_ type is error:" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::ResizeT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new attr value failed";
      return RET_ERROR;
    }
    if (prim.instance_name() == "ResizeNearestNeighbor") {
      attr->method = schema::ResizeMethod_NEAREST;
    } else if (prim.instance_name() == "ResizeBilinear") {
      attr->method = schema::ResizeMethod_LINEAR;
    } else {
      delete attr;
      MS_LOG(ERROR) << "wrong resize type";
      return RET_ERROR;
    }
    std::vector<int> targetSize = CastToInt(prim.GetAttr("size"));
    attr->newHeight = targetSize.at(0);
    attr->newWidth = targetSize.at(1);
    attr->alignCorners = GetValue<bool>(prim.GetAttr("align_corners"));
    if (attr->alignCorners) {
      attr->coordinateTransformMode = schema::CoordinateTransformMode_ALIGN_CORNERS;
    }
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      if (attr != nullptr) {
        delete attr;
      }
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else

int Resize::GetFormat() const { return this->primitive_->value_as_Resize()->format(); }
int Resize::GetMethod() const { return this->primitive_->value_as_Resize()->method(); }
int64_t Resize::GetNewHeight() const { return this->primitive_->value_as_Resize()->newHeight(); }
int64_t Resize::GetNewWidth() const { return this->primitive_->value_as_Resize()->newWidth(); }
int Resize::GetCoordinateTransformMode() const {
  return this->primitive_->value_as_Resize()->coordinateTransformMode();
}
bool Resize::GetPreserveAspectRatio() const { return this->primitive_->value_as_Resize()->preserveAspectRatio(); }
int Resize::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Resize();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Resize return nullptr";
    return RET_ERROR;
  }
  auto val_offset =
    schema::CreateResize(*fbb, attr->format(), attr->method(), attr->newHeight(), attr->newWidth(),
                         attr->alignCorners(), attr->preserveAspectRatio(), attr->coordinateTransformMode(),
                         attr->cubicCoeff(), attr->excludeOutside(), attr->extrapolationValue(), attr->nearestMode());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Resize, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *ResizeCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Resize>(primitive); }
Registry ResizeRegistry(schema::PrimitiveType_Resize, ResizeCreator);
#endif

namespace {
constexpr int kInputRank = 4;
}  // namespace
int64_t Resize::new_height() const { return new_height_; }
int64_t Resize::new_width() const { return new_width_; }
int Resize::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  if (input == nullptr) {
    return RET_ERROR;
  }
  if (!input->shape().empty() && input->shape().size() != kInputRank) {
    MS_LOG(ERROR) << "Size of input shape is wrong.";
    return RET_ERROR;
  }

  auto output = outputs_.front();
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  std::vector<int> output_shape;
  output_shape.push_back(input->Batch());
  auto ret = CalculateNewHeightAndWidth(inputs_);
  if (ret == RET_OK) {
    output_shape.push_back(new_height_);
    output_shape.push_back(new_width_);
    output_shape.push_back(input->Channel());
    output->set_shape(output_shape);
  }
  return ret;
}

int Resize::CalculateNewHeightAndWidth(const std::vector<lite::Tensor *> &inputs) {
  auto input = inputs.front();
  if (inputs.size() == kDoubleNum) {
    auto shape_tensor = inputs.at(1);
    if (shape_tensor->data_c() == nullptr) {
      MS_LOG(INFO) << "Do infer shape in runtime.";
      return RET_INFER_INVALID;
    }
    size_t shape_size = shape_tensor->ElementsNum();
    switch (shape_size) {
      case kQuadrupleNum: {
        if (shape_tensor->data_type() == kNumberTypeInt32) {
          auto data = reinterpret_cast<int32_t *>(shape_tensor->data_c());
          if (data == nullptr) {
            MS_LOG(INFO) << "Resize op size can't cast int.";
            return RET_INFER_INVALID;
          }
          switch (shape_tensor->format()) {
            case schema::Format_NCHW:
              new_height_ = data[2];
              new_width_ = data[3];
              break;
            case schema::Format_NHWC:
              new_height_ = data[1];
              new_width_ = data[2];
              break;
            default:
              MS_LOG(INFO) << "Resize don't support tensor format.";
              return RET_INFER_INVALID;
          }
        } else if (shape_tensor->data_type() == kNumberTypeFloat32) {
          auto data = reinterpret_cast<float *>(shape_tensor->data_c());
          if (data == nullptr) {
            MS_LOG(INFO) << "Resize op size can't cast float.";
            return RET_INFER_INVALID;
          }
          switch (shape_tensor->format()) {
            case schema::Format_NCHW:
              new_height_ = data[2] * input->Height();
              new_width_ = data[3] * input->Width();
              break;
            case schema::Format_NHWC:
              new_height_ = data[1] * input->Height();
              new_width_ = data[2] * input->Width();
              break;
            default:
              MS_LOG(INFO) << "Resize don't support tensor format.";
              return RET_INFER_INVALID;
          }
        }
        break;
      }
      case kDoubleNum: {
        auto data = reinterpret_cast<int32_t *>(shape_tensor->data_c());
        if (data == nullptr) {
          MS_LOG(INFO) << "Resize op size can't cast float.";
          return RET_INFER_INVALID;
        }
        new_height_ = data[0];
        new_width_ = data[1];
        break;
      }
      case kSingleNum: {
        // caffe zoom_factor
        int scale;
        if (shape_tensor->data_type() == kNumberTypeInt32) {
          auto data = reinterpret_cast<int *>(shape_tensor->data_c());
          if (data == nullptr) {
            MS_LOG(INFO) << "Resize op size can't cast int.";
            return RET_INFER_INVALID;
          }
          scale = data[0];
        } else {
          MS_LOG(ERROR) << "Unsupported data type:" << shape_tensor->data_type();
          return RET_INFER_ERR;
        }
        new_height_ = input->Height() + (input->Height() - 1) * (scale - 1);
        new_width_ = input->Width() + (input->Width() - 1) * (scale - 1);
        break;
      }
      default: {
        MS_LOG(ERROR) << "Unsupported shape size:" << shape_size;
        return RET_INFER_ERR;
      }
    }
  } else if (inputs.size() == kSingleNum) {
    new_height_ = GetNewHeight();
    new_width_ = GetNewWidth();
  } else if (inputs.size() == kQuadrupleNum) {
    if (inputs[3]->data_c() == nullptr) {
      return RET_INFER_INVALID;
    }
    new_height_ = static_cast<int *>(inputs.at(3)->data_c())[0];
    new_height_ = static_cast<int *>(inputs.at(3)->data_c())[1];
  } else {
    MS_LOG(ERROR) << "inputs tensor size invalid.";
    return RET_INFER_ERR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
