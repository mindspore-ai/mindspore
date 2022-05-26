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

#include <algorithm>
#include <utility>
#include "src/extendrt/delegate/tensorrt/op/slice_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
namespace {
class StrideSliceTensorRTUtil final : public SliceTensorRTUtil {
 public:
  StrideSliceTensorRTUtil() = default;
  ~StrideSliceTensorRTUtil() = default;
  bool IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                 const std::vector<mindspore::MSTensor> &out_tensors) override {
    if (in_tensors.size() < HAS_AXIS - 1) {
      MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
      return false;
    }
    if (out_tensors.size() != 1) {
      MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
      return false;
    }
    if (in_tensors.at(BEGINS_INDEX).Data() == nullptr || in_tensors.at(ENDS_INDEX).Data() == nullptr) {
      MS_LOG(ERROR) << "invalid input tensor for: " << op_name_;
      return false;
    }
    return true;
  }
  std::tuple<nvinfer1::Dims, nvinfer1::Dims, nvinfer1::Dims> GetSliceParams(
    const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
    const std::vector<mindspore::MSTensor> &out_tensors) override {
    const mindspore::MSTensor &begin = in_tensors.at(BEGINS_INDEX);
    const mindspore::MSTensor &stride = in_tensors.back();
    const mindspore::MSTensor &end = in_tensors.at(ENDS_INDEX);

    nvinfer1::Dims start_dims;
    nvinfer1::Dims size_dims;
    nvinfer1::Dims stride_dims;

    size_t axis_index = in_tensors.size() == HAS_AXIS ? AXIS_INDEX : -1;
    auto out_shape = out_tensors.front().Shape();
    if (static_cast<size_t>(begin.ElementNum()) == in_tensors.at(0).Shape().size()) {
      start_dims = lite::ConvertCudaDims(begin.Data().get(), begin.ElementNum());
      if (shrink_axis_ == 0) {
        size_dims = lite::ConvertCudaDims(out_shape);
      } else {
        size_dims.nbDims = start_dims.nbDims;
        auto end_dims = lite::ConvertCudaDims(end.Data().get(), end.ElementNum());
        for (int i = 0; i < size_dims.nbDims; i++) {
          size_dims.d[i] = end_dims.d[i] - start_dims.d[i];
        }
      }
      stride_dims = lite::ConvertCudaDims(stride.Data().get(), stride.ElementNum());
    } else {
      if (axis_index == -1 || in_tensors.at(axis_index).ElementNum() != 1) {
        MS_LOG(ERROR) << "invalid input params for " << op_name_;
        return {};
      }
      int axis_value = *(static_cast<const int *>(in_tensors.at(axis_index).Data().get()));
      int start_value = *(static_cast<const int *>(begin.Data().get()));
      start_dims.nbDims = in_tensors.at(0).Shape().size();
      for (int i = 0; i < start_dims.nbDims; i++) {
        start_dims.d[i] = (i == axis_value) ? start_value : 0;
      }

      size_dims = lite::ConvertCudaDims(out_shape);
      int stride_value = *(static_cast<const int *>(stride.Data().get()));
      stride_dims = nvinfer1::Dims{size_dims.nbDims, {}};
      std::fill(stride_dims.d, stride_dims.d + stride_dims.nbDims, stride_value);
    }
    return std::make_tuple(start_dims, size_dims, stride_dims);
  }
  nvinfer1::ITensor *PostProcess(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *input,
                                 const std::vector<mindspore::MSTensor> &in_tensors,
                                 const std::vector<mindspore::MSTensor> &out_tensors) {
    if (shrink_axis_ != 0) {
      return Reshape(network, input, out_tensors.at(0).Shape());
    }
    return input;
  }
  void SetShrinkAxis(int shrink_axis) { shrink_axis_ = shrink_axis; }

 private:
  int shrink_axis_;
};

class SliceFusionTensorRTUtil final : public SliceTensorRTUtil {
 public:
  SliceFusionTensorRTUtil() = default;
  ~SliceFusionTensorRTUtil() = default;
  bool IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                 const std::vector<mindspore::MSTensor> &out_tensors) override {
    if (in_tensors.size() != SLICE_INPUT_SIZE) {
      MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
      return false;
    }
    if (out_tensors.size() != 1) {
      MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
      return false;
    }
    return true;
  }
  std::tuple<nvinfer1::Dims, nvinfer1::Dims, nvinfer1::Dims> GetSliceParams(
    const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
    const std::vector<mindspore::MSTensor> &out_tensors) override {
    const auto &input = in_tensors.at(0);
    const auto &begin = in_tensors.at(1);
    const auto &size = in_tensors.at(SIZE_INDEX);

    auto start_dims = lite::ConvertCudaDims(begin.Data().get(), begin.ElementNum());
    auto size_dims = lite::ConvertCudaDims(size.Data().get(), size.ElementNum());
    auto stride_dims = lite::ConvertCudaDims(1, begin.ElementNum());

    return std::make_tuple(start_dims, size_dims, stride_dims);
  }
};

class CropTensorRTUtil final : public SliceTensorRTUtil {
 public:
  CropTensorRTUtil() = default;
  ~CropTensorRTUtil() = default;
  bool IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                 const std::vector<mindspore::MSTensor> &out_tensors) override {
    if (in_tensors.size() != CROP_INPUT_SIZE) {
      MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
      return false;
    }
    if (out_tensors.size() != 1) {
      MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
      return false;
    }
    auto crop_primitive = primitive->value_as_Crop();
    if (crop_primitive == nullptr) {
      MS_LOG(ERROR) << "Cast primitive to crop fail";
      return false;
    }
    axis_ = static_cast<int>(crop_primitive->axis());
    auto offsets_ptr = crop_primitive->offsets();
    if (offsets_ptr == nullptr) {
      MS_LOG(ERROR) << "Crop Op do not have offset attr";
      return false;
    }
    if (axis_ < 0) {
      axis_ += in_tensors.at(0).Shape().size();
    }
    if (axis_ < 0 || axis_ + offsets_ptr->size() != in_tensors.at(0).Shape().size()) {
      MS_LOG(ERROR) << "axis and offsets not match input tensor shape, axis is " << crop_primitive->axis()
                    << " , offsets size is " << offsets_ptr->size() << " , input size is "
                    << in_tensors.at(0).Shape().size();
      return false;
    }
    if (in_tensors.at(0).Shape().size() != in_tensors.at(1).Shape().size()) {
      MS_LOG(ERROR) << "input tensor 0 and 1 size not equal,"
                    << " input 0 size is " << in_tensors.at(0).Shape().size() << " , input tensor 1 size is "
                    << in_tensors.at(1).Shape().size();
      return false;
    }
    return true;
  }
  std::tuple<nvinfer1::Dims, nvinfer1::Dims, nvinfer1::Dims> GetSliceParams(
    const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
    const std::vector<mindspore::MSTensor> &out_tensors) override {
    auto crop_primitive = primitive->value_as_Crop();
    auto offsets_ptr = crop_primitive->offsets();

    std::vector<int> begin(in_tensors.at(0).Shape().size(), 0);
    for (size_t i = 0; i != offsets_ptr->size(); ++i) {
      begin[axis_ + i] = offsets_ptr->Get(i);
    }

    std::vector<int> size(in_tensors.at(0).Shape().size());
    for (size_t i = 0; i != size.size(); ++i) {
      size[i] = in_tensors.at(1).Shape().at(i);
    }

    auto start_dims = lite::ConvertCudaDims(&begin[0], begin.size());
    auto size_dims = lite::ConvertCudaDims(&size[0], size.size());
    auto stride_dims = lite::ConvertCudaDims(1, begin.size());

    return std::make_tuple(start_dims, size_dims, stride_dims);
  }

 private:
  int axis_;
};
}  // namespace

SliceTensorRT::SliceTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                             const schema::QuantType &quant_type)
    : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {
  if (primitive->value_type() == schema::PrimitiveType_StridedSlice) {
    auto slice_fusion_util = std::make_unique<StrideSliceTensorRTUtil>();
    slice_fusion_util->SetShrinkAxis(primitive->value_as_StridedSlice()->shrink_axis_mask());
    util_ = std::move(slice_fusion_util);
  } else if (primitive->value_type() == schema::PrimitiveType_SliceFusion) {
    util_ = std::make_unique<SliceFusionTensorRTUtil>();
  } else if (primitive->value_type() == schema::PrimitiveType_Crop) {
    util_ = std::make_unique<CropTensorRTUtil>();
  } else {
    util_ = nullptr;
  }
  if (util_ != nullptr) {
    util_->op_name_ = op_name_;
  }
}

int SliceTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                             const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (util_ == nullptr) {
    MS_LOG(ERROR) << "Unsupported op_type: " << op_name_;
    return RET_ERROR;
  }
  if (!util_->IsSupport(primitive, in_tensors, out_tensors)) {
    return RET_ERROR;
  }
  dynamic_shape_params_.support_dynamic_ = false;
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
}

int SliceTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  ITensorHelper slice_input;
  int ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[0], &slice_input);
  if (ret != RET_OK || slice_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim input tensor failed for " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::Dims start_dims;
  nvinfer1::Dims size_dims;
  nvinfer1::Dims stride_dims;
  std::tie(start_dims, size_dims, stride_dims) = util_->GetSliceParams(op_primitive_, in_tensors_, out_tensors_);
  if (start_dims.nbDims == -1 || size_dims.nbDims == -1 || stride_dims.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::ISliceLayer *slice_layer = network->addSlice(*slice_input.trt_tensor_, start_dims, size_dims, stride_dims);
  if (slice_layer == nullptr) {
    MS_LOG(ERROR) << "add Slice op failed for TensorRT: " << op_name_;
    return RET_ERROR;
  }
  this->layer_ = slice_layer;
  slice_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = slice_layer->getOutput(0);
  out_tensor = util_->PostProcess(network, out_tensor, in_tensors_, out_tensors_);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "output tensor create failed";
    return RET_ERROR;
  }
  out_tensor->setName((op_name_ + "_output").c_str());

  this->AddInnerOutTensors(ITensorHelper{out_tensor, slice_input.format_, slice_input.same_format_});
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_StridedSlice, SliceTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_SliceFusion, SliceTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Crop, SliceTensorRT)
}  // namespace mindspore::lite
