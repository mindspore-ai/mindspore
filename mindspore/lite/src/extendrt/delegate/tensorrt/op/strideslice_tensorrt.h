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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_STRIDE_SLICE_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_STRIDE_SLICE_TENSORRT_H_
#include <string>
#include <vector>
#include <tuple>
#include <memory>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
constexpr int BEGINS_INDEX = 1;
constexpr int ENDS_INDEX = 2;
constexpr int HAS_AXIS = 5;
constexpr int AXIS_INDEX = 3;
class StrideSliceTensorRT : public TensorRTOp {
 public:
  StrideSliceTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                      const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~StrideSliceTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;

 private:
  nvinfer1::ITensor *GetDynamicSliceSize(TensorRTContext *ctx, nvinfer1::ITensor *input, int, int);
  nvinfer1::ILayer *MakeLayer(TensorRTContext *ctx, const ITensorHelper &slice_input);
  size_t shrink_axis_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_STRIDE_SLICE_TENSORRT_H_
