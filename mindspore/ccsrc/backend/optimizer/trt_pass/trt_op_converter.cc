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

#include <vector>
#include <string>
#include <NvInfer.h>
#include "backend/optimizer/trt_pass/trt_converter_context.h"
#include "backend/optimizer/trt_pass/trt_op_factory.h"
#include "backend/kernel_compiler/gpu/trt/trt_utils.h"

namespace mindspore {
namespace opt {
namespace {
ConvertResult AddReshapeLayer(AnfNodePtr node, std::shared_ptr<TrtConverterContext> context) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1 || !inputs[0].IsTensor()) {
    MS_LOG(ERROR) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  auto *layer = context->network()->addShuffle(*inputs[0].tensor());
  MS_EXCEPTION_IF_NULL(layer);

  const auto &input_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  const auto &output_shape = AnfAlgo::GetOutputInferShape(node, 0);
  if (input_shape[0] != output_shape[0]) {
    MS_LOG(ERROR) << "Reshape does not support modify batch size. Input batch size: " << input_shape[0]
                  << "Output batch size: " << output_shape[0];
    return {false, {}};
  }

  const nvinfer1::Dims &dims = TrtUtils::MsDimsToTrtDims(output_shape, false);
  layer->setReshapeDimensions(dims);

  return {true, {LayerInput(layer->getOutput(0))}};
}

ConvertResult AddElementLayer(AnfNodePtr node, std::shared_ptr<TrtConverterContext> context,
                              nvinfer1::ElementWiseOperation op_type) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 2) {
    MS_LOG(ERROR) << "Input num not match: " << inputs.size() << ", with 2 expected.";
    return {false, {}};
  }

  const std::vector<size_t> &x1_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  const std::vector<size_t> &x2_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
  const std::vector<size_t> &y_shape = AnfAlgo::GetOutputInferShape(node, 0);

  // Keep to output
  auto Broadcast = [&context, &y_shape](nvinfer1::ITensor *tensor, const std::vector<size_t> &x_shape) {
    if (x_shape.size() == y_shape.size()) {
      return tensor;
    }

    // Copy x_shape to dim with tail align, and fill left axis with 1.
    // For example:
    //    x: [C, H, W]
    //    y: [N, C, H, W]
    //  dim: [1, C, H, W]
    nvinfer1::Dims dim;
    dim.nbDims = SizeToInt(y_shape.size());
    std::fill(dim.d, dim.d + dim.nbDims, 1);
    size_t offset = y_shape.size() - x_shape.size();
    for (size_t i = 0; i < x_shape.size(); i++) {
      dim.d[i + offset] = SizeToInt(x_shape[i]);
    }

    auto *layer = context->network()->addShuffle(*tensor);
    MS_EXCEPTION_IF_NULL(layer);
    layer->setReshapeDimensions(dim);

    return layer->getOutput(0);
  };

  auto *x1 = Broadcast(inputs[0].tensor(), x1_shape);
  auto *x2 = Broadcast(inputs[1].tensor(), x2_shape);
  auto *layer = context->network()->addElementWise(*x1, *x2, op_type);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {LayerInput(layer->getOutput(0))}};
}

ConvertResult AddPoolingLayer(AnfNodePtr node, std::shared_ptr<TrtConverterContext> context,
                              nvinfer1::PoolingType pooling_type) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1 || !inputs[0].IsTensor()) {
    MS_LOG(ERROR) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  const auto &format = AnfAlgo::GetNodeAttr<std::string>(node, "format");
  if (format != "NCHW") {
    MS_LOG(ERROR) << "The format: " << format << " not supported.";
    return {false, {}};
  }

  const auto &kernel_size = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "kernel_size");
  auto *layer = context->network()->addPoolingNd(
    *(inputs[0].tensor()), pooling_type, nvinfer1::DimsHW{LongToInt(kernel_size[2]), LongToInt(kernel_size[3])});
  MS_EXCEPTION_IF_NULL(layer);

  const auto &strides = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "strides");
  layer->setStride(nvinfer1::DimsHW{LongToInt(strides[2]), LongToInt(strides[3])});

  auto pad_mode = AnfAlgo::GetNodeAttr<std::string>(node, "pad_mode");
  std::transform(pad_mode.begin(), pad_mode.end(), pad_mode.begin(), toupper);
  if (pad_mode == "SAME") {
    layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  }

  return {true, {LayerInput(layer->getOutput(0))}};
}

ConvertResult AddActivationLayer(AnfNodePtr node, std::shared_ptr<TrtConverterContext> context,
                                 nvinfer1::ActivationType act_type) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1 || !inputs[0].IsTensor()) {
    MS_LOG(ERROR) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  auto *layer = context->network()->addActivation(*inputs[0].tensor(), act_type);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {LayerInput(layer->getOutput(0))}};
}
}  // namespace

// Register operator converter from AnfNode to trt layer: `OPNAME` should keep the same as primitive definition.
#define MS_TRT_CONVERTER_FUNC_REG(OPNAME)                                                                 \
  ConvertResult Gpu##OPNAME##TrtConverter(AnfNodePtr node, std::shared_ptr<TrtConverterContext> context); \
  static const TrtOpRegister(Gpu##OPNAME##ConverterRegister)(#OPNAME, Gpu##OPNAME##TrtConverter);         \
  ConvertResult Gpu##OPNAME##TrtConverter(AnfNodePtr node, std::shared_ptr<TrtConverterContext> context)

MS_TRT_CONVERTER_FUNC_REG(Conv2D) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 2 || !inputs[0].IsTensor() || !inputs[1].IsWeight()) {
    MS_LOG(ERROR) << "Input num not match: " << inputs.size() << ", with 2 expected.";
    return {false, {}};
  }

  const auto &data_format = AnfAlgo::GetNodeAttr<std::string>(node, "format");
  if (data_format != "NCHW") {
    MS_LOG(ERROR) << "The format: " << data_format << " not supported.";
    return {false, {}};
  }

  const auto &kernel_size = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "kernel_size");
  const auto &out_channel = AnfAlgo::GetNodeAttr<int64_t>(node, "out_channel");
  nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
  auto *layer = context->network()->addConvolutionNd(
    *(inputs[0].tensor()), LongToInt(out_channel),
    nvinfer1::DimsHW{LongToInt(kernel_size[0]), LongToInt(kernel_size[1])}, *(inputs[1].weight()), bias);
  MS_EXCEPTION_IF_NULL(layer);

  const auto &strides = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "stride");
  layer->setStride(nvinfer1::DimsHW{LongToInt(strides[2]), LongToInt(strides[3])});

  auto pad_mode = AnfAlgo::GetNodeAttr<std::string>(node, "pad_mode");
  std::transform(pad_mode.begin(), pad_mode.end(), pad_mode.begin(), toupper);
  if (pad_mode == "SAME") {
    layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  }

  if (pad_mode == "PAD") {
    const auto &pad_list = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "pad_list");
    layer->setPrePadding(nvinfer1::DimsHW{LongToInt(pad_list[0]), LongToInt(pad_list[2])});
    layer->setPostPadding(nvinfer1::DimsHW{LongToInt(pad_list[1]), LongToInt(pad_list[3])});
  }

  return {true, {LayerInput(layer->getOutput(0))}};
}

// Binary broadcast operators.
MS_TRT_CONVERTER_FUNC_REG(Add) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kSUM); }
MS_TRT_CONVERTER_FUNC_REG(Sub) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kSUB); }
MS_TRT_CONVERTER_FUNC_REG(Mul) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kPROD); }
MS_TRT_CONVERTER_FUNC_REG(Div) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kDIV); }
MS_TRT_CONVERTER_FUNC_REG(Pow) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kPOW); }
MS_TRT_CONVERTER_FUNC_REG(Maximum) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kMAX); }
MS_TRT_CONVERTER_FUNC_REG(Minimum) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kMIN); }
MS_TRT_CONVERTER_FUNC_REG(FloorDiv) {
  return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kFLOOR_DIV);
}

// Pooling operators.
MS_TRT_CONVERTER_FUNC_REG(AvgPool) { return AddPoolingLayer(node, context, nvinfer1::PoolingType::kAVERAGE); }
MS_TRT_CONVERTER_FUNC_REG(MaxPool) { return AddPoolingLayer(node, context, nvinfer1::PoolingType::kMAX); }

// Activation operators.
MS_TRT_CONVERTER_FUNC_REG(ReLU) { return AddActivationLayer(node, context, nvinfer1::ActivationType::kRELU); }
MS_TRT_CONVERTER_FUNC_REG(Sigmoid) { return AddActivationLayer(node, context, nvinfer1::ActivationType::kSIGMOID); }
MS_TRT_CONVERTER_FUNC_REG(Tanh) { return AddActivationLayer(node, context, nvinfer1::ActivationType::kTANH); }
MS_TRT_CONVERTER_FUNC_REG(Elu) { return AddActivationLayer(node, context, nvinfer1::ActivationType::kELU); }
MS_TRT_CONVERTER_FUNC_REG(Softsign) { return AddActivationLayer(node, context, nvinfer1::ActivationType::kSOFTSIGN); }

MS_TRT_CONVERTER_FUNC_REG(GeLU) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1) {
    MS_LOG(ERROR) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  const std::vector<size_t> &x_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  nvinfer1::Dims dim;
  dim.nbDims = SizeToInt(x_shape.size());
  std::fill(dim.d, dim.d + dim.nbDims, 1);

  auto AddConst = [&context, &dim](const float &coeff) -> nvinfer1::ITensor * {
    std::shared_ptr<tensor::Tensor> weight = context->CreateTempWeight(kNumberTypeFloat32, {1});
    auto value = static_cast<float *>(weight->data_c());
    value[0] = coeff;

    auto *layer = context->network()->addConstant(dim, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, value, 1});
    MS_EXCEPTION_IF_NULL(layer);
    return layer->getOutput(0);
  };

  // y = 0.5 * x * (1 + tanh(0.7978846 * (x + 0.044715 * x^3)))
  auto *c1 = AddConst(0.5f);
  auto *c2 = AddConst(1.0f);
  auto *c3 = AddConst(0.7978846f);
  auto *c4 = AddConst(0.044715f);
  auto *c5 = AddConst(3.0f);

  auto *x = inputs[0].tensor();
  nvinfer1::ILayer *layer = context->network()->addElementWise(*x, *c5, nvinfer1::ElementWiseOperation::kPOW);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*c4, *layer->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*x, *layer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*c3, *layer->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addActivation(*layer->getOutput(0), nvinfer1::ActivationType::kTANH);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*c2, *layer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*x, *layer->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*c1, *layer->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {LayerInput(layer->getOutput(0))}};
}

MS_TRT_CONVERTER_FUNC_REG(MatMul) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 2 || !inputs[0].IsTensor() || !inputs[1].IsWeight()) {
    MS_LOG(ERROR) << "Input num not match: " << inputs.size() << ", with 2 expected.";
    return {false, {}};
  }

  const auto &transpose_a = AnfAlgo::GetNodeAttr<bool>(node, "transpose_a");
  const auto &transpose_b = AnfAlgo::GetNodeAttr<bool>(node, "transpose_b");
  if (inputs[0].IsTensor() && inputs[1].IsWeight() && !transpose_a && transpose_b) {
    // Reshape x from (M, K) to (M, K, 1, 1)
    nvinfer1::Dims unsqueeze_dims = inputs[0].tensor()->getDimensions();
    for (size_t i = 0; i < 2; i++) {
      unsqueeze_dims.d[unsqueeze_dims.nbDims++] = 1;
    }
    auto x_reshape = context->network()->addShuffle(*inputs[0].tensor());
    x_reshape->setReshapeDimensions(unsqueeze_dims);

    // Apply addFullyConnected: y = x * w^T + b
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    const auto &w_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
    auto *layer =
      context->network()->addFullyConnected(*x_reshape->getOutput(0), w_shape[0], *inputs[1].weight(), bias);
    MS_EXCEPTION_IF_NULL(layer);

    // Reshape x from (M, N, 1, 1) to (M, N)
    const auto &y_shape = AnfAlgo::GetOutputInferShape(node, 0);
    const nvinfer1::Dims &y_dims = TrtUtils::MsDimsToTrtDims(y_shape, false);
    auto *squeeze_y = context->network()->addShuffle(*layer->getOutput(0));
    squeeze_y->setReshapeDimensions(y_dims);

    return {true, {LayerInput(squeeze_y->getOutput(0))}};
  } else {
    // convert weight to tensor and appy addMatrixMultiply
    MS_LOG(ERROR) << "Operator not implemented: " << node->DebugString();
    return {false, {}};
  }
}

MS_TRT_CONVERTER_FUNC_REG(BiasAdd) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 2 || !inputs[0].IsTensor() || !inputs[1].IsWeight()) {
    MS_LOG(ERROR) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  const auto &x_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  const auto &bias_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
  const auto &format = AnfAlgo::GetNodeAttr<std::string>(node, "format");
  const string::size_type &pos = format.find("C");
  if (pos == std::string::npos || pos >= x_shape.size()) {
    MS_LOG(ERROR) << "The format " << format << "' invalid";
    return {false, {}};
  }

  // Convert Weight to ITensor which
  nvinfer1::Dims unsqueeze_bias_dims;
  unsqueeze_bias_dims.nbDims = x_shape.size();
  std::fill(unsqueeze_bias_dims.d, unsqueeze_bias_dims.d + unsqueeze_bias_dims.nbDims, 1);
  unsqueeze_bias_dims.d[pos] = SizeToInt(bias_shape[0]);
  nvinfer1::ITensor *bias = context->network()->addConstant(unsqueeze_bias_dims, *inputs[1].weight())->getOutput(0);

  // Create Broadcast Add layer.
  auto *layer = context->network()->addElementWise(*inputs[0].tensor(), *bias, nvinfer1::ElementWiseOperation::kSUM);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {LayerInput(layer->getOutput(0))}};
}

MS_TRT_CONVERTER_FUNC_REG(Reshape) { return AddReshapeLayer(node, context); }

MS_TRT_CONVERTER_FUNC_REG(ExpandDims) { return AddReshapeLayer(node, context); }

MS_TRT_CONVERTER_FUNC_REG(Squeeze) { return AddReshapeLayer(node, context); }

MS_TRT_CONVERTER_FUNC_REG(BatchNorm) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 5 || !inputs[0].IsTensor() || !inputs[1].IsWeight() || !inputs[2].IsWeight() ||
      !inputs[3].IsWeight() || !inputs[4].IsWeight()) {
    MS_LOG(ERROR) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  auto primitive = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  auto is_training = AnfAlgo::GetNodeAttr<bool>(node, "is_training");
  if (is_training != false) {
    MS_LOG(ERROR) << "Operation not support, is_training: " << is_training;
    return {false, {}};
  }

  const auto &format = AnfAlgo::GetNodeAttr<std::string>(node, "format");
  if (format != "NCHW") {
    MS_LOG(ERROR) << "The format " << format << "' invalid";
    return {false, {}};
  }

  // scale = gamma / sqrt(var + epsilon)
  // y = (x - mean) * scale + beta
  //   = x * scale - mean * scale + beta
  //   = x * coeff + bias
  auto gamma = static_cast<const float *>(inputs[1].weight()->values);
  auto beta = static_cast<const float *>(inputs[2].weight()->values);
  auto mean = static_cast<const float *>(inputs[3].weight()->values);
  auto var = static_cast<const float *>(inputs[4].weight()->values);
  auto epsilon = AnfAlgo::GetNodeAttr<float>(node, "epsilon");

  const TypeId &type = AnfAlgo::GetPrevNodeOutputInferDataType(node, 1);
  const std::vector<size_t> &shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
  int64_t channel_num = SizeToLong(shape[0]);
  auto coeff = context->CreateTempWeight(type, shape);
  auto bias = context->CreateTempWeight(type, shape);
  auto coeff_value = static_cast<float *>(coeff->data_c());
  auto bias_value = static_cast<float *>(bias->data_c());
  for (int64_t i = 0; i < channel_num; i++) {
    float scale = gamma[i] / sqrtf(var[i] + epsilon);
    coeff_value[i] = scale;
    bias_value[i] = beta[i] - mean[i] * scale;
  }

  const nvinfer1::Weights &scale{nvinfer1::DataType::kFLOAT, coeff_value, channel_num};
  const nvinfer1::Weights &shift{nvinfer1::DataType::kFLOAT, bias_value, channel_num};
  const nvinfer1::Weights &pow{nvinfer1::DataType::kFLOAT, nullptr, 0};
  auto *layer = context->network()->addScale(*inputs[0].tensor(), nvinfer1::ScaleMode::kCHANNEL, shift, scale, pow);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {LayerInput(layer->getOutput(0))}};
}

MS_TRT_CONVERTER_FUNC_REG(Concat) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() == 0) {
    MS_LOG(ERROR) << "Get inputs failed. Input num: " << inputs.size();
    return {false, {}};
  }

  std::vector<nvinfer1::ITensor *> tensors;
  for (const auto &input : inputs) {
    if (input.IsWeight()) {
      MS_LOG(ERROR) << "Concat input do not support weight.";
      return {false, {}};
    }
    tensors.push_back(input.tensor());
  }

  auto *layer = context->network()->addConcatenation(tensors.data(), tensors.size());
  MS_EXCEPTION_IF_NULL(layer);

  auto axis = static_cast<int>(AnfAlgo::GetNodeAttr<int64_t>(node, "axis"));
  if (axis < 0) {
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
    axis += SizeToInt(input_shape.size());
  }
  layer->setAxis(axis);

  return {true, {LayerInput(layer->getOutput(0))}};
}

MS_TRT_CONVERTER_FUNC_REG(Conv2DBackpropInput) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 2 || !inputs[0].IsTensor() || !inputs[1].IsWeight()) {
    MS_LOG(ERROR) << "Input num not match: " << inputs.size() << ", with 2 expected.";
    return {false, {}};
  }

  const auto &format = AnfAlgo::GetNodeAttr<std::string>(node, "format");
  if (format != "NCHW") {
    MS_LOG(ERROR) << "The format: " << format << " not supported.";
    return {false, {}};
  }

  const auto &kernel_size = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "kernel_size");
  const auto &output_shape = AnfAlgo::GetOutputInferShape(node, 0);
  const nvinfer1::Weights &bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
  auto *layer = context->network()->addDeconvolutionNd(
    *(inputs[0].tensor()), SizeToInt(output_shape[1]),
    nvinfer1::DimsHW{LongToInt(kernel_size[0]), LongToInt(kernel_size[1])}, *(inputs[1].weight()), bias);
  MS_EXCEPTION_IF_NULL(layer);

  const auto &strides = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "stride");
  layer->setStride(nvinfer1::DimsHW{LongToInt(strides[2]), LongToInt(strides[3])});

  auto pad_mode = AnfAlgo::GetNodeAttr<std::string>(node, "pad_mode");
  std::transform(pad_mode.begin(), pad_mode.end(), pad_mode.begin(), toupper);
  if (pad_mode == "SAME") {
    layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  }

  if (pad_mode == "PAD") {
    const auto &pad_list = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "pad_list");
    layer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);
    layer->setPrePadding(nvinfer1::DimsHW{LongToInt(pad_list[0]), LongToInt(pad_list[2])});
    layer->setPostPadding(nvinfer1::DimsHW{LongToInt(pad_list[1]), LongToInt(pad_list[3])});
  }

  return {true, {LayerInput(layer->getOutput(0))}};
}
}  // namespace opt
}  // namespace mindspore
