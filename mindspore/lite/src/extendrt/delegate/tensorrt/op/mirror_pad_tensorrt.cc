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

#include "src/extendrt/delegate/tensorrt/op/mirror_pad_tensorrt.h"
#include "ops/mirror_pad.h"
namespace mindspore::lite {
constexpr int SIZE_INDEX = 2;
int MirrorPadTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                 const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int MirrorPadTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  auto mirrorpad_op = AsOps<ops::MirrorPad>();
  CHECK_NULL_RETURN(mirrorpad_op);
  auto pad_mode = mirrorpad_op->get_mode();
  int ret = RunAsTrtOps(ctx, pad_mode);
  return ret;
}

int MirrorPadTensorRT::RunAsTrtOps(TensorRTContext *ctx, string mode) {
  auto pad_input = input(ctx, 0).trt_tensor_;
  if (pad_input == nullptr) {
    MS_LOG(ERROR) << "TensorRt Tensor of input 0 of pad " << op_name_ << " is nullptr";
    return RET_ERROR;
  }
  auto input_shape = ConvertMSShape(pad_input->getDimensions());
  if (!in_tensors_[1].IsConst()) {
    MS_LOG(ERROR) << "Input 1 of pad " << op_name_ << " is not constant";
    return RET_ERROR;
  }
  auto pad_vec = ConvertTensorAsIntVector(in_tensors_[1]);
  if (pad_vec.empty()) {
    MS_LOG(ERROR) << "Failed to get pad input, node: " << op_name_;
    return RET_ERROR;
  }
  constexpr size_t pad_multi_times = 2;
  if (pad_vec.size() % pad_multi_times != 0 && pad_vec.size() != input_shape.size() * pad_multi_times) {
    MS_LOG(ERROR) << "pad tensor is invalid, pad count: " << pad_vec.size()
                  << ", input dims count: " << input_shape.size() << ", op: " << op_name_;
    return RET_ERROR;
  }
#if TRT_VERSION_GE(8, 0)
  std::vector<int32_t> start_values;
  std::vector<int32_t> size_values;
  std::vector<int64_t> stride_values;
  for (size_t i = 0; i < pad_vec.size(); i += pad_multi_times) {
    start_values.push_back(-pad_vec[i]);
    stride_values.push_back(1);
    size_values.push_back(pad_vec[i] + pad_vec[i + 1]);
  }
  nvinfer1::ITensor *size;
  auto totalPadding = ctx->ConvertTo1DTensor(size_values);
  auto shape = ctx->network()->addShape(*pad_input)->getOutput(0);
  size = ctx->network()->addElementWise(*shape, *totalPadding, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  auto slice_layer =
    ctx->network()->addSlice(*pad_input, ConvertCudaDims(start_values), {}, ConvertCudaDims(stride_values));
  slice_layer->setInput(SIZE_INDEX, *size);
  if (slice_layer == nullptr) {
    MS_LOG(ERROR) << "Failed to add slice layer for op " << op_name_;
    return RET_ERROR;
  }
  if (mode == "REFLECT") {
    slice_layer->setMode(nvinfer1::SliceMode::kREFLECT);
  } else {
    MS_LOG(ERROR) << "Not support padding mode " << mode << " for op " << op_name_;
    return RET_ERROR;
  }
  slice_layer->setName(op_name_.c_str());
  this->layer_ = slice_layer;
  auto out_tensor = slice_layer->getOutput(0);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to get output tensor of op " << op_name_;
    return RET_ERROR;
  }
  auto output_tensor = ITensorHelper{out_tensor, NCHW, true};
  ctx->RegisterTensor(output_tensor, out_tensors_[0].Name());
  return RET_OK;
#else
  MS_LOG(ERROR) << "Only support pad mode constant and input dims count 8 when trt version < 8.0, op:  " << op_name_;
  return RET_ERROR;
#endif
}
REGISTER_TENSORRT_CREATOR(ops::kNameMirrorPad, MirrorPadTensorRT)
}  // namespace mindspore::lite
