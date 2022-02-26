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
#ifndef MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_SHUFFLE_TENSORRT_H_
#define MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_SHUFFLE_TENSORRT_H_
#include <string>
#include <vector>
#include "src/delegate/tensorrt/op/tensorrt_op.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
class ShuffleTensorRT : public TensorRTOp {
 public:
  ShuffleTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                  const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                  const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~ShuffleTensorRT() override = default;

  int AddInnerOp(nvinfer1::INetworkDefinition *network) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

 private:
  int InputTensorPreprocess();
  int AddSqueezeOp(nvinfer1::IShuffleLayer *shuffle_layer);
  int AddUnsqueezeOp(nvinfer1::IShuffleLayer *shuffle_layer);
  int AddTransposeOp(nvinfer1::IShuffleLayer *shuffle_layer);
  int AddReshapeOp(nvinfer1::IShuffleLayer *shuffle_layer);
  int AddFlattenOp(nvinfer1::IShuffleLayer *shuffle_layer);
  int AddExpandDimsOp(nvinfer1::IShuffleLayer *shuffle_layer);
  nvinfer1::ITensor *ExpandDim(nvinfer1::IShuffleLayer *shuffle_layer, nvinfer1::ITensor *input_tensor, int axis);
  nvinfer1::Dims InferReshapeDims(const nvinfer1::Dims &input_dims, const std::vector<int64_t> &ms_input_shape,
                                  const std::vector<int64_t> &ms_output_shape);

  Format out_format_ = Format::NHWC;
  nvinfer1::ITensor *shuffler_input_{nullptr};
  nvinfer1::ITensor *shuffler_output_{nullptr};
  nvinfer1::INetworkDefinition *network_{nullptr};
  const flatbuffers::Vector<int64_t> *param_axis_{nullptr};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_SHUFFLE_TENSORRT_H_
