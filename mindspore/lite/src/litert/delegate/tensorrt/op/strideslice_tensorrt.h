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
#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_TENSORRT_OP_STRIDE_SLICE_TENSORRT_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_TENSORRT_OP_STRIDE_SLICE_TENSORRT_H_
#include <string>
#include <vector>
#include <tuple>
#include <memory>
#include "src/litert/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
constexpr int BEGINS_INDEX = 1;
constexpr int ENDS_INDEX = 2;
constexpr int HAS_AXIS = 5;
constexpr int AXIS_INDEX = 3;
class StrideSliceTensorRT : public TensorRTOp {
 public:
  StrideSliceTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                      const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                      const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~StrideSliceTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

 private:
  nvinfer1::ITensor *GetDynamicSliceSize(TensorRTContext *ctx, nvinfer1::ITensor *input,
                                         const nvinfer1::Dims &size_dims);
  nvinfer1::ITensor *GetDynamicAxisSliceSize(TensorRTContext *ctx, nvinfer1::ITensor *input, int size_dim, int axis,
                                             nvinfer1::ITensor *size_tensor);
  int ComputeSliceDims(TensorRTContext *ctx, ITensorHelper *slice_input);
  size_t shrink_axis_;
  size_t start_axis_;
  size_t end_axis_;
  nvinfer1::Dims start_dims_;
  nvinfer1::Dims size_dims_;
  nvinfer1::Dims stride_dims_;
  nvinfer1::ITensor *size_tensor_{nullptr};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_TENSORRT_OP_STRIDE_SLICE_TENSORRT_H_
