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
#include "ops/strided_slice.h"
#include "ops/fusion/slice_fusion.h"
#include "ops/crop.h"

namespace mindspore::lite {
namespace {
class StrideSliceTensorRTUtil final : public SliceTensorRTUtil {
 public:
  StrideSliceTensorRTUtil() = default;
  ~StrideSliceTensorRTUtil() = default;
  bool IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                 const std::vector<TensorInfo> &out_tensors) override {
    if (in_tensors.size() < HAS_AXIS - 1) {
      MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
      return false;
    }
    if (out_tensors.size() != 1) {
      MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
      return false;
    }
    if (!in_tensors.at(BEGINS_INDEX).IsConst() || !in_tensors.at(ENDS_INDEX).IsConst()) {
      MS_LOG(ERROR) << "invalid input tensor for: " << op_name_;
      return false;
    }
    return true;
  }
  std::tuple<nvinfer1::Dims, nvinfer1::Dims, nvinfer1::Dims> GetSliceParams(const BaseOperatorPtr &base_operator,
                                                                            const std::vector<TensorInfo> &in_tensors,
                                                                            const std::vector<TensorInfo> &out_tensors,
                                                                            const ITensorHelper &helper) override {
    const TensorInfo &begin = in_tensors.at(BEGINS_INDEX);
    const TensorInfo &stride = in_tensors.back();
    const TensorInfo &end = in_tensors.at(ENDS_INDEX);
    nvinfer1::Dims start_dims;
    nvinfer1::Dims size_dims;
    nvinfer1::Dims stride_dims;

    int64_t axis_index = in_tensors.size() == HAS_AXIS ? AXIS_INDEX : -1;
    if (begin.ElementNum() == helper.trt_tensor_->getDimensions().nbDims) {
      start_dims = lite::ConvertCudaDims(begin);
      size_dims.nbDims = start_dims.nbDims;
      auto end_dims = lite::ConvertCudaDims(end);
      for (int i = 0; i < size_dims.nbDims; i++) {
        size_dims.d[i] = end_dims.d[i] - start_dims.d[i];
      }
      stride_dims = lite::ConvertCudaDims(stride);
    } else {
      if (axis_index == -1 || in_tensors.at(axis_index).ElementNum() != 1) {
        MS_LOG(ERROR) << "invalid input params for " << op_name_;
        return {};
      }
      int axis_value = *(static_cast<const int *>(in_tensors.at(axis_index).Data()));
      int start_value = *(static_cast<const int *>(begin.Data()));
      int size_value = *(static_cast<const int *>(end.Data()));
      auto input_dims = helper.trt_tensor_->getDimensions();
      start_dims.nbDims = input_dims.nbDims;
      size_dims.nbDims = input_dims.nbDims;
      for (int i = 0; i < start_dims.nbDims; i++) {
        if (i == axis_value) {
          start_dims.d[i] = start_value;
          size_dims.d[i] = size_value >= 0 ? std::min(size_value, input_dims.d[i]) - start_dims.d[i]
                                           : size_value + input_dims.d[i] - start_dims.d[i];
        } else {
          start_dims.d[i] = 0;
          size_dims.d[i] = helper.trt_tensor_->getDimensions().d[i];
        }
      }

      int stride_value = *(static_cast<const int *>(stride.Data()));
      stride_dims = nvinfer1::Dims{size_dims.nbDims, {}};
      std::fill(stride_dims.d, stride_dims.d + stride_dims.nbDims, stride_value);
    }
    return std::make_tuple(start_dims, size_dims, stride_dims);
  }
  nvinfer1::ITensor *PostProcess(TensorRTContext *ctx, nvinfer1::ITensor *input,
                                 const std::vector<TensorInfo> &in_tensors,
                                 const std::vector<TensorInfo> &out_tensors) override {
    if (shrink_axis_ != 0) {
      auto shape = ConvertMSShape(input->getDimensions());
      for (int i = shape.size() - 1; i >= 0; --i) {
        int mask = 1 << i;
        if ((shrink_axis_ & mask) != 0) {
          shape.erase(shape.begin() + i);
        }
      }
      return Reshape(ctx, input, shape);
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
  bool IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                 const std::vector<TensorInfo> &out_tensors) override {
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
  std::tuple<nvinfer1::Dims, nvinfer1::Dims, nvinfer1::Dims> GetSliceParams(const BaseOperatorPtr &base_operator,
                                                                            const std::vector<TensorInfo> &in_tensors,
                                                                            const std::vector<TensorInfo> &out_tensors,
                                                                            const ITensorHelper &helper) override {
    const auto &begin = in_tensors.at(1);
    const auto &size = in_tensors.at(SIZE_INDEX);

    auto start_dims = lite::ConvertCudaDims(begin.Data(), begin.ElementNum());
    auto size_dims = lite::ConvertCudaDims(size.Data(), size.ElementNum());
    for (int i = 0; i != size_dims.nbDims; ++i) {
      if (size_dims.d[i] == -1) {
        size_dims.d[i] = helper.trt_tensor_->getDimensions().d[i];
      }
    }
    auto stride_dims = lite::ConvertCudaDims(1, begin.ElementNum());

    return std::make_tuple(start_dims, size_dims, stride_dims);
  }
};

class CropTensorRTUtil final : public SliceTensorRTUtil {
 public:
  CropTensorRTUtil() = default;
  ~CropTensorRTUtil() = default;
  bool IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                 const std::vector<TensorInfo> &out_tensors) override {
    if (in_tensors.size() != CROP_INPUT_SIZE) {
      MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
      return false;
    }
    if (out_tensors.size() != 1) {
      MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
      return false;
    }
    auto crop_primitive = TensorRTOp::AsOps<ops::Crop>(base_operator);
    if (crop_primitive == nullptr) {
      MS_LOG(ERROR) << "Cast primitive to crop fail";
      return false;
    }
    axis_ = static_cast<int>(crop_primitive->get_axis());
    return true;
  }
  std::tuple<nvinfer1::Dims, nvinfer1::Dims, nvinfer1::Dims> GetSliceParams(const BaseOperatorPtr &base_operator,
                                                                            const std::vector<TensorInfo> &in_tensors,
                                                                            const std::vector<TensorInfo> &out_tensors,
                                                                            const ITensorHelper &helper) override {
    auto crop_primitive = TensorRTOp::AsOps<ops::Crop>(base_operator);
    auto offsets_ptr = crop_primitive->get_offsets();
    if (offsets_ptr.empty()) {
      MS_LOG(ERROR) << "Crop Op do not have offset attr";
      return {};
    }
    if (axis_ < 0) {
      axis_ += helper.trt_tensor_->getDimensions().nbDims;
    }
    if (axis_ < 0 || axis_ + SizeToInt(offsets_ptr.size()) != helper.trt_tensor_->getDimensions().nbDims) {
      MS_LOG(ERROR) << "axis and offsets not match input tensor shape, axis is " << crop_primitive->get_axis()
                    << " , offsets size is " << offsets_ptr.size() << " , input size is "
                    << helper.trt_tensor_->getDimensions().nbDims;
      return {};
    }

    std::vector<int> begin(helper.trt_tensor_->getDimensions().nbDims, 0);
    for (size_t i = 0; i != offsets_ptr.size(); ++i) {
      begin[axis_ + i] = offsets_ptr[i];
    }

    std::vector<int> size(helper.trt_tensor_->getDimensions().nbDims);
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

SliceTensorRT::SliceTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                             const std::vector<TensorInfo> &out_tensors, std::string name)
    : TensorRTOp(base_operator, in_tensors, out_tensors, name) {
  if (type_ == ops::kNameStridedSlice) {
    auto slice_fusion_util = std::make_unique<StrideSliceTensorRTUtil>();
    auto op = AsOps<ops::StridedSlice>();
    slice_fusion_util->SetShrinkAxis(op->get_shrink_axis_mask());
    util_ = std::move(slice_fusion_util);
  } else if (type_ == ops::kNameSliceFusion) {
    util_ = std::make_unique<SliceFusionTensorRTUtil>();
  } else if (type_ == ops::kNameCrop) {
    util_ = std::make_unique<CropTensorRTUtil>();
  } else {
    util_ = nullptr;
  }
  if (util_ != nullptr) {
    util_->op_name_ = op_name_;
  }
}

int SliceTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                             const std::vector<TensorInfo> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (util_ == nullptr) {
    MS_LOG(ERROR) << "Unsupported op_type: " << op_name_;
    return RET_ERROR;
  }
  if (!util_->IsSupport(base_operator, in_tensors, out_tensors)) {
    return RET_ERROR;
  }
  dynamic_shape_params_.support_dynamic_ = false;
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
}

int SliceTensorRT::AddInnerOp(TensorRTContext *ctx) {
  ITensorHelper slice_input;
  int ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), &slice_input);
  if (ret != RET_OK || slice_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim input tensor failed for " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::Dims start_dims;
  nvinfer1::Dims size_dims;
  nvinfer1::Dims stride_dims;
  std::tie(start_dims, size_dims, stride_dims) =
    util_->GetSliceParams(base_operator_, in_tensors_, out_tensors_, slice_input);
  if (start_dims.nbDims == -1 || size_dims.nbDims == -1 || stride_dims.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::ISliceLayer *slice_layer =
    ctx->network()->addSlice(*slice_input.trt_tensor_, start_dims, size_dims, stride_dims);
  if (slice_layer == nullptr) {
    MS_LOG(ERROR) << "add Slice op failed for TensorRT: " << op_name_;
    return RET_ERROR;
  }
  this->layer_ = slice_layer;
  slice_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = slice_layer->getOutput(0);
  out_tensor = util_->PostProcess(ctx, out_tensor, in_tensors_, out_tensors_);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "output tensor create failed";
    return RET_ERROR;
  }
  auto helper = ITensorHelper{out_tensor, slice_input.format_, slice_input.same_format_};
  ctx->RegisterTensor(helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "slice output : " << GetTensorFormat(helper);
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameStridedSlice, SliceTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameSliceFusion, SliceTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameCrop, SliceTensorRT)
}  // namespace mindspore::lite
