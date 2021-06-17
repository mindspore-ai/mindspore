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
#include <variant>
#include <NvInfer.h>
#include "backend/optimizer/trt_pass/trt_converter_context.h"
#include "backend/optimizer/trt_pass/trt_op_factory.h"
#include "backend/kernel_compiler/gpu/trt/trt_utils.h"

namespace mindspore {
namespace opt {
namespace {
nvinfer1::ITensor *ToShape(LayerInput *input, const std::vector<size_t> &shape,
                           std::shared_ptr<TrtConverterContext> context) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(context);

  if (!input->IsTensor()) {
    MS_LOG(WARNING) << "Expect Tensor but got weight";
    return nullptr;
  }

  const nvinfer1::Dims &src_dim = input->tensor()->getDimensions();
  const nvinfer1::Dims &dst_dim = TrtUtils::MsDimsToTrtDims(shape, false);
  if (TrtUtils::IsSameShape(src_dim, dst_dim)) {
    return input->tensor();
  }

  auto *layer = context->network()->addShuffle(*input->tensor());
  MS_EXCEPTION_IF_NULL(layer);
  layer->setReshapeDimensions(dst_dim);

  return layer->getOutput(0);
}

nvinfer1::ITensor *ToTensor(LayerInput *input, const std::vector<size_t> &shape,
                            std::shared_ptr<TrtConverterContext> context) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(context);
  if (input->IsTensor()) {
    return ToShape(input, shape, context);
  }

  const nvinfer1::Dims &dim = TrtUtils::MsDimsToTrtDims(shape, false);
  auto *const_layer = context->network()->addConstant(dim, *input->weight());
  MS_EXCEPTION_IF_NULL(const_layer);
  return const_layer->getOutput(0);
}

ConvertResult AddReshapeLayer(AnfNodePtr node, std::shared_ptr<TrtConverterContext> context) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1 || !inputs[0].IsTensor()) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  auto *layer = context->network()->addShuffle(*inputs[0].tensor());
  MS_EXCEPTION_IF_NULL(layer);
  const auto &output_shape = AnfAlgo::GetOutputInferShape(node, 0);
  const nvinfer1::Dims &dims = TrtUtils::MsDimsToTrtDims(output_shape, false);
  layer->setReshapeDimensions(dims);

  return {true, {layer->getOutput(0)}};
}

ConvertResult AddElementLayer(AnfNodePtr node, std::shared_ptr<TrtConverterContext> context,
                              nvinfer1::ElementWiseOperation op_type) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 2) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 2 expected.";
    return {false, {}};
  }

  const std::vector<size_t> &x1_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  const std::vector<size_t> &x2_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
  const std::vector<size_t> &y_shape = AnfAlgo::GetOutputInferShape(node, 0);

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

  auto *x1 = Broadcast(ToTensor(&inputs[0], x1_shape, context), x1_shape);
  auto *x2 = Broadcast(ToTensor(&inputs[1], x2_shape, context), x2_shape);
  auto *layer = context->network()->addElementWise(*x1, *x2, op_type);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {layer->getOutput(0)}};
}

ConvertResult AddPoolingLayer(AnfNodePtr node, std::shared_ptr<TrtConverterContext> context,
                              nvinfer1::PoolingType pooling_type) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1 || !inputs[0].IsTensor()) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  const auto &format = AnfAlgo::GetNodeAttr<std::string>(node, "format");
  if (format != "NCHW") {
    MS_LOG(WARNING) << "The format: " << format << " not supported.";
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

  return {true, {layer->getOutput(0)}};
}

ConvertResult AddActivationLayer(AnfNodePtr node, std::shared_ptr<TrtConverterContext> context,
                                 nvinfer1::ActivationType act_type) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1 || !inputs[0].IsTensor()) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  auto *layer = context->network()->addActivation(*inputs[0].tensor(), act_type);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {layer->getOutput(0)}};
}

ConvertResult AddUnaryLayer(AnfNodePtr node, std::shared_ptr<TrtConverterContext> context,
                            nvinfer1::UnaryOperation op_type) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 2 expected.";
    return {false, {}};
  }

  auto *layer = context->network()->addUnary(*inputs[0].tensor(), op_type);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {layer->getOutput(0)}};
}

ConvertResult AddReduceLayer(AnfNodePtr node, std::shared_ptr<TrtConverterContext> context,
                             nvinfer1::ReduceOperation op_type) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 2 expected.";
    return {false, {}};
  }

  // Calculate reduce axes bitmask
  const std::vector<size_t> &input_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  const ValuePtr &value = AnfAlgo::GetCNodePrimitive(node)->GetAttr("axis");
  uint32_t reduce_axes = 0;
  if (value->isa<ValueTuple>() || value->isa<ValueList>()) {
    const auto &axis = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "axis");
    for (size_t i = 0; i < axis.size(); i++) {
      int offset = axis[i] >= 0 ? LongToInt(axis[i]) : LongToInt(axis[i] + input_shape.size());
      reduce_axes |= 1UL << offset;
    }
  } else {
    const auto &axis = AnfAlgo::GetNodeAttr<int64_t>(node, "axis");
    int offset = axis >= 0 ? LongToInt(axis) : LongToInt(axis + input_shape.size());
    reduce_axes = 1UL << offset;
  }

  // Tensor-RT do not support reduce with no dimensions.
  // Skip reduce operator if reduce_axes == 0
  if (reduce_axes == 0) {
    MS_LOG(WARNING) << "No dimension be be reduced. " << node->DebugString();
    return {true, {inputs[0].tensor()}};
  }

  bool keep_dims = AnfAlgo::GetNodeAttr<bool>(node, "keep_dims");
  // Tensor-RT do not support reduce all dimensions with keep_dims == false.
  // Reduce with keep_dims = true, add apply reshape latter.
  bool post_reshape = false;
  if (keep_dims == false && (reduce_axes == (1UL << input_shape.size()) - 1)) {
    keep_dims = true;
    post_reshape = true;
  }

  nvinfer1::IReduceLayer *layer = context->network()->addReduce(*inputs[0].tensor(), op_type, reduce_axes, keep_dims);
  MS_EXCEPTION_IF_NULL(layer);

  if (post_reshape) {
    nvinfer1::IShuffleLayer *reshape_layer = context->network()->addShuffle(*layer->getOutput(0));
    MS_EXCEPTION_IF_NULL(reshape_layer);

    nvinfer1::Dims dim;
    dim.nbDims = 1;
    dim.d[0] = 1;
    reshape_layer->setReshapeDimensions(dim);

    return {true, {reshape_layer->getOutput(0)}};
  }

  return {true, {layer->getOutput(0)}};
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
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 2 expected.";
    return {false, {}};
  }

  const auto &data_format = AnfAlgo::GetNodeAttr<std::string>(node, "format");
  if (data_format != "NCHW") {
    MS_LOG(WARNING) << "The format: " << data_format << " not supported.";
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

  const auto &group = AnfAlgo::GetNodeAttr<int64_t>(node, "group");
  layer->setNbGroups(SizeToInt(group));

  return {true, {layer->getOutput(0)}};
}

// Binary broadcast operators.
MS_TRT_CONVERTER_FUNC_REG(Add) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kSUM); }
MS_TRT_CONVERTER_FUNC_REG(Sub) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kSUB); }
MS_TRT_CONVERTER_FUNC_REG(Mul) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kPROD); }
MS_TRT_CONVERTER_FUNC_REG(Div) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kDIV); }
MS_TRT_CONVERTER_FUNC_REG(RealDiv) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kDIV); }
MS_TRT_CONVERTER_FUNC_REG(Pow) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kPOW); }
MS_TRT_CONVERTER_FUNC_REG(Maximum) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kMAX); }
MS_TRT_CONVERTER_FUNC_REG(Minimum) { return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kMIN); }
MS_TRT_CONVERTER_FUNC_REG(FloorDiv) {
  return AddElementLayer(node, context, nvinfer1::ElementWiseOperation::kFLOOR_DIV);
}

// Unary operators
MS_TRT_CONVERTER_FUNC_REG(Exp) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kEXP); }
MS_TRT_CONVERTER_FUNC_REG(Log) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kLOG); }
MS_TRT_CONVERTER_FUNC_REG(Sqrt) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kSQRT); }
MS_TRT_CONVERTER_FUNC_REG(Reciprocal) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kRECIP); }
MS_TRT_CONVERTER_FUNC_REG(Abs) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kABS); }
MS_TRT_CONVERTER_FUNC_REG(Neg) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kNEG); }
MS_TRT_CONVERTER_FUNC_REG(Sin) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kSIN); }
MS_TRT_CONVERTER_FUNC_REG(Cos) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kCOS); }
MS_TRT_CONVERTER_FUNC_REG(Tan) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kTAN); }
MS_TRT_CONVERTER_FUNC_REG(Sinh) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kSINH); }
MS_TRT_CONVERTER_FUNC_REG(Cosh) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kCOSH); }
MS_TRT_CONVERTER_FUNC_REG(Asin) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kASIN); }
MS_TRT_CONVERTER_FUNC_REG(Acos) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kACOS); }
MS_TRT_CONVERTER_FUNC_REG(Atan) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kATAN); }
MS_TRT_CONVERTER_FUNC_REG(Asinh) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kASINH); }
MS_TRT_CONVERTER_FUNC_REG(Acosh) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kACOSH); }
MS_TRT_CONVERTER_FUNC_REG(Ceil) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kCEIL); }
MS_TRT_CONVERTER_FUNC_REG(Floor) { return AddUnaryLayer(node, context, nvinfer1::UnaryOperation::kFLOOR); }

// Reduce operators
MS_TRT_CONVERTER_FUNC_REG(ReduceSum) { return AddReduceLayer(node, context, nvinfer1::ReduceOperation::kSUM); }
MS_TRT_CONVERTER_FUNC_REG(ReduceMean) { return AddReduceLayer(node, context, nvinfer1::ReduceOperation::kAVG); }
MS_TRT_CONVERTER_FUNC_REG(ReduceMax) { return AddReduceLayer(node, context, nvinfer1::ReduceOperation::kMAX); }
MS_TRT_CONVERTER_FUNC_REG(ReduceMin) { return AddReduceLayer(node, context, nvinfer1::ReduceOperation::kMIN); }
MS_TRT_CONVERTER_FUNC_REG(ReduceProd) { return AddReduceLayer(node, context, nvinfer1::ReduceOperation::kPROD); }

// Pooling operators.
MS_TRT_CONVERTER_FUNC_REG(AvgPool) { return AddPoolingLayer(node, context, nvinfer1::PoolingType::kAVERAGE); }
MS_TRT_CONVERTER_FUNC_REG(MaxPool) { return AddPoolingLayer(node, context, nvinfer1::PoolingType::kMAX); }

// Activation operators.
MS_TRT_CONVERTER_FUNC_REG(ReLU) { return AddActivationLayer(node, context, nvinfer1::ActivationType::kRELU); }
MS_TRT_CONVERTER_FUNC_REG(Sigmoid) { return AddActivationLayer(node, context, nvinfer1::ActivationType::kSIGMOID); }
MS_TRT_CONVERTER_FUNC_REG(Tanh) { return AddActivationLayer(node, context, nvinfer1::ActivationType::kTANH); }
MS_TRT_CONVERTER_FUNC_REG(Elu) { return AddActivationLayer(node, context, nvinfer1::ActivationType::kELU); }
MS_TRT_CONVERTER_FUNC_REG(Softsign) { return AddActivationLayer(node, context, nvinfer1::ActivationType::kSOFTSIGN); }

MS_TRT_CONVERTER_FUNC_REG(ReLU6) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
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

  // y = max(0.0, min(6.0, x)
  auto *c0 = AddConst(0.0f);
  auto *c1 = AddConst(6.0f);
  auto *x = inputs[0].tensor();
  nvinfer1::ILayer *layer = context->network()->addElementWise(*x, *c1, nvinfer1::ElementWiseOperation::kMIN);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*layer->getOutput(0), *c0, nvinfer1::ElementWiseOperation::kMAX);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(GeLU) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
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

  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(HSigmoid) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
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

  // y = max(0, min(1.0, (x + 3.0)/6.0))
  auto *c0 = AddConst(0.0f);
  auto *c1 = AddConst(1.0f);
  auto *c2 = AddConst(3.0f);
  auto *c3 = AddConst(6.0f);
  auto *x = inputs[0].tensor();
  nvinfer1::ILayer *layer = context->network()->addElementWise(*x, *c2, nvinfer1::ElementWiseOperation::kSUM);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*layer->getOutput(0), *c3, nvinfer1::ElementWiseOperation::kDIV);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*layer->getOutput(0), *c1, nvinfer1::ElementWiseOperation::kMIN);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*layer->getOutput(0), *c0, nvinfer1::ElementWiseOperation::kMAX);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(HSwish) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
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

  // y = x * Relu6(x + 3.0) / 6.0
  // Relu6(x) = min(max(x, 0.0), 6.0)
  auto *c0 = AddConst(0.0f);
  auto *c1 = AddConst(3.0f);
  auto *c2 = AddConst(6.0f);
  auto *x = inputs[0].tensor();
  nvinfer1::ILayer *layer = context->network()->addElementWise(*x, *c1, nvinfer1::ElementWiseOperation::kSUM);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*layer->getOutput(0), *c0, nvinfer1::ElementWiseOperation::kMAX);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*layer->getOutput(0), *c2, nvinfer1::ElementWiseOperation::kMIN);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*layer->getOutput(0), *c2, nvinfer1::ElementWiseOperation::kDIV);
  MS_EXCEPTION_IF_NULL(layer);
  layer = context->network()->addElementWise(*x, *layer->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(MatMul) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 2) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 2 expected.";
    return {false, {}};
  }

  const auto &transpose_a = AnfAlgo::GetNodeAttr<bool>(node, "transpose_a");
  const auto &transpose_b = AnfAlgo::GetNodeAttr<bool>(node, "transpose_b");
  if (inputs[0].IsTensor() && inputs[1].IsWeight() && transpose_a == false && transpose_b == true) {
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

    return {true, {squeeze_y->getOutput(0)}};
  } else {
    auto op1 = transpose_a ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;
    auto op2 = transpose_b ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;
    const std::vector<size_t> &x1_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
    const std::vector<size_t> &x2_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
    nvinfer1::ITensor *x1 = ToTensor(&inputs[0], x1_shape, context);
    nvinfer1::ITensor *x2 = ToTensor(&inputs[1], x2_shape, context);
    auto *layer = context->network()->addMatrixMultiply(*x1, op1, *x2, op2);
    MS_EXCEPTION_IF_NULL(layer);
    return {true, {layer->getOutput(0)}};
  }
}

MS_TRT_CONVERTER_FUNC_REG(BatchMatMul) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 2) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 2 expected.";
    return {false, {}};
  }

  const auto &transpose_a = AnfAlgo::GetNodeAttr<bool>(node, "transpose_a");
  const auto &transpose_b = AnfAlgo::GetNodeAttr<bool>(node, "transpose_b");
  const auto &trt_transpose1 = transpose_a ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;
  const auto &trt_transpose2 = transpose_b ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;

  std::vector<size_t> shape1 = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  std::vector<size_t> shape2 = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
  nvinfer1::ITensor *tensor1 = ToTensor(&inputs[0], shape1, context);
  nvinfer1::ITensor *tensor2 = ToTensor(&inputs[1], shape2, context);
  auto *layer = context->network()->addMatrixMultiply(*tensor1, trt_transpose1, *tensor2, trt_transpose2);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(BiasAdd) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 2) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  const auto &x_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  const auto &bias_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
  const auto &format = AnfAlgo::GetNodeAttr<std::string>(node, "format");
  const string::size_type &pos = format.find("C");
  if (pos == std::string::npos || pos >= x_shape.size()) {
    MS_LOG(WARNING) << "The format " << format << "' invalid";
    return {false, {}};
  }

  // Convert bias to ITensor same dims as x.
  std::vector<size_t> unsqueeze_bias_dims(x_shape.size(), 1);
  unsqueeze_bias_dims[pos] = SizeToInt(bias_shape[0]);
  nvinfer1::ITensor *bias = ToTensor(&inputs[1], unsqueeze_bias_dims, context);

  // Create Broadcast Add layer.
  auto *layer = context->network()->addElementWise(*inputs[0].tensor(), *bias, nvinfer1::ElementWiseOperation::kSUM);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {layer->getOutput(0)}};
}

// NoOp
MS_TRT_CONVERTER_FUNC_REG(Reshape) { return AddReshapeLayer(node, context); }
MS_TRT_CONVERTER_FUNC_REG(ExpandDims) { return AddReshapeLayer(node, context); }
MS_TRT_CONVERTER_FUNC_REG(Squeeze) { return AddReshapeLayer(node, context); }
MS_TRT_CONVERTER_FUNC_REG(Flatten) { return AddReshapeLayer(node, context); }

MS_TRT_CONVERTER_FUNC_REG(BatchNorm) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 5 || !inputs[0].IsTensor() || !inputs[1].IsWeight() || !inputs[2].IsWeight() ||
      !inputs[3].IsWeight() || !inputs[4].IsWeight()) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  auto primitive = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  auto is_training = AnfAlgo::GetNodeAttr<bool>(node, "is_training");
  if (is_training != false) {
    MS_LOG(WARNING) << "Operation not support, is_training: " << is_training;
    return {false, {}};
  }

  const auto &format = AnfAlgo::GetNodeAttr<std::string>(node, "format");
  if (format != "NCHW") {
    MS_LOG(WARNING) << "The format " << format << "' invalid";
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

  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(Concat) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() == 0) {
    MS_LOG(WARNING) << "Get inputs failed. Input num: " << inputs.size();
    return {false, {}};
  }

  std::vector<nvinfer1::ITensor *> tensors;
  for (const auto &input : inputs) {
    if (input.IsWeight()) {
      MS_LOG(WARNING) << "Concat input do not support weight.";
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

  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(Conv2DBackpropInput) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 2 || !inputs[0].IsTensor() || !inputs[1].IsWeight()) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 2 expected.";
    return {false, {}};
  }

  const auto &format = AnfAlgo::GetNodeAttr<std::string>(node, "format");
  if (format != "NCHW") {
    MS_LOG(WARNING) << "The format: " << format << " not supported.";
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

  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(Slice) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1 || !inputs[0].IsTensor()) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  const auto &begin = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "begin");
  const auto &size = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "size");

  nvinfer1::Dims trt_start = TrtUtils::MsDimsToTrtDims(begin, false);
  nvinfer1::Dims trt_size = TrtUtils::MsDimsToTrtDims(size, false);
  nvinfer1::Dims trt_stride;
  for (int32_t i = 0; i < trt_start.nbDims; i++) {
    trt_stride.d[trt_stride.nbDims++] = 1;
  }

  auto *layer = context->network()->addSlice(*inputs[0].tensor(), trt_start, trt_size, trt_stride);
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(Transpose) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1 || !inputs[0].IsTensor()) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  const auto &perm = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "perm");
  nvinfer1::Permutation trt_perm;
  for (size_t i = 0; i < perm.size(); i++) {
    trt_perm.order[i] = LongToInt(perm[i]);
  }

  auto *layer = context->network()->addShuffle(*inputs[0].tensor());
  MS_EXCEPTION_IF_NULL(layer);
  layer->setFirstTranspose(trt_perm);

  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(Softmax) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1 || !inputs[0].IsTensor()) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  const std::vector<size_t> &input_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  const ValuePtr &value = AnfAlgo::GetCNodePrimitive(node)->GetAttr("axis");
  uint32_t reduce_axes = 0;
  if (value->isa<ValueTuple>() || value->isa<ValueList>()) {
    const auto &axis = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "axis");
    if (axis.size() != 1) {
      MS_LOG(WARNING) << "Only one axis can be set. Axis size" << axis.size();
      return {false, {}};
    }
    int offset = axis[0] >= 0 ? LongToInt(axis[0]) : LongToInt(axis[0] + input_shape.size());
    reduce_axes = 1U << offset;
  } else {
    const auto &axis = AnfAlgo::GetNodeAttr<int64_t>(node, "axis");
    int offset = axis >= 0 ? LongToInt(axis) : LongToInt(axis + input_shape.size());
    reduce_axes = 1UL << offset;
  }

  auto *layer = context->network()->addSoftMax(*inputs[0].tensor());
  MS_EXCEPTION_IF_NULL(layer);
  layer->setAxes(reduce_axes);
  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(LogSoftmax) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1 || !inputs[0].IsTensor()) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 1 expected.";
    return {false, {}};
  }

  const std::vector<size_t> &input_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  const auto &axis = AnfAlgo::GetNodeAttr<int64_t>(node, "axis");
  int offset = axis >= 0 ? LongToInt(axis) : LongToInt(axis + input_shape.size());
  uint32_t reduce_axes = 1UL << offset;

  auto *softmax_layer = context->network()->addSoftMax(*inputs[0].tensor());
  MS_EXCEPTION_IF_NULL(softmax_layer);
  softmax_layer->setAxes(reduce_axes);

  auto *log_layer = context->network()->addUnary(*softmax_layer->getOutput(0), nvinfer1::UnaryOperation::kLOG);
  MS_EXCEPTION_IF_NULL(log_layer);

  return {true, {log_layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(Gather) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 2) {
    MS_LOG(WARNING) << "Input num not match: " << inputs.size() << ", with 2 expected.";
    return {false, {}};
  }

  const std::vector<size_t> &input_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  auto axis = AnfAlgo::GetNodeAttr<int64_t>(node, "axis");
  axis = axis >= 0 ? axis : axis + input_shape.size();

  nvinfer1::ITensor *input = ToTensor(&inputs[0], input_shape, context);
  const std::vector<size_t> &indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
  nvinfer1::ITensor *indices = ToTensor(&inputs[1], indices_shape, context);

  auto *layer = context->network()->addGather(*input, *indices, LongToInt(axis));
  MS_EXCEPTION_IF_NULL(layer);

  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(Cast) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 1) {
    MS_LOG(WARNING) << "Get inputs failed. Input num: " << inputs.size();
    return {false, {}};
  }

  const std::vector<size_t> &input_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  nvinfer1::ITensor *input = ToTensor(&inputs[0], input_shape, context);

  const TypeId &dst_type = AnfAlgo::GetOutputInferDataType(node, 0);
  std::variant<bool, nvinfer1::DataType> type = TrtUtils::MsDtypeToTrtDtype(dst_type);
  if (type.index() != 1) {
    return {false, {}};
  }
  auto trt_type = std::get<nvinfer1::DataType>(type);
  auto *layer = context->network()->addIdentity(*input);
  layer->setOutputType(0, trt_type);

  if (trt_type == nvinfer1::DataType::kHALF) {
    MS_LOG(WARNING) << "The model is exported with auto-mixed-precsion or manual precision mode. "
                    << "Retreat inference with native backend. It is recommended that export FP32 model "
                    << "and then inference with FP16 precision mode configuration.";
    return {false, {}};
  }
  return {true, {layer->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(LayerNorm) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret || inputs.size() != 3 || !inputs[0].IsTensor()) {
    MS_LOG(WARNING) << "Get inputs failed. Input num: " << inputs.size();
    return {false, {}};
  }

  // Calculate reduce axes
  const std::vector<size_t> &input_shape = AnfAlgo::GetOutputInferShape(node, 0);
  auto begin_norm_axis = AnfAlgo::GetNodeAttr<int64_t>(node, "begin_norm_axis");
  begin_norm_axis = begin_norm_axis >= 0 ? begin_norm_axis : begin_norm_axis + input_shape.size();
  uint32_t reduce_axes = 0;
  for (size_t i = LongToSize(begin_norm_axis); i < input_shape.size(); i++) {
    reduce_axes |= 1UL << i;
  }

  // Reshape gamma and beta for broadcast
  auto begin_params_axis = AnfAlgo::GetNodeAttr<int64_t>(node, "begin_params_axis");
  begin_params_axis = begin_params_axis >= 0 ? begin_params_axis : begin_params_axis + input_shape.size();
  std::vector<size_t> param_shape = input_shape;
  for (size_t j = 0; j < LongToSize(begin_params_axis); j++) {
    param_shape[j] = 1;
  }

  auto epsilon = AnfAlgo::GetNodeAttr<float>(node, "epsilon");
  std::shared_ptr<tensor::Tensor> weight = context->CreateTempWeight(kNumberTypeFloat32, {1});
  auto value = static_cast<float *>(weight->data_c());
  value[0] = epsilon;
  nvinfer1::Dims dim;
  dim.nbDims = SizeToInt(input_shape.size());
  for (size_t i = 0; i < input_shape.size(); i++) {
    dim.d[i] = 1;
  }
  auto *epsilon_layer = context->network()->addConstant(dim, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, value, 1});
  MS_EXCEPTION_IF_NULL(epsilon_layer);

  // y = (x - mean) / sqrt(var) * gamma + beta
  auto *mean = context->network()->addReduce(*inputs[0].tensor(), nvinfer1::ReduceOperation::kAVG, reduce_axes, true);
  MS_EXCEPTION_IF_NULL(mean);
  auto *sub =
    context->network()->addElementWise(*inputs[0].tensor(), *mean->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
  MS_EXCEPTION_IF_NULL(sub);
  auto *pow =
    context->network()->addElementWise(*sub->getOutput(0), *sub->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
  MS_EXCEPTION_IF_NULL(pow);
  auto *var = context->network()->addReduce(*pow->getOutput(0), nvinfer1::ReduceOperation::kAVG, reduce_axes, true);
  MS_EXCEPTION_IF_NULL(var);
  auto *var_epsilon = context->network()->addElementWise(*var->getOutput(0), *epsilon_layer->getOutput(0),
                                                         nvinfer1::ElementWiseOperation::kSUM);
  MS_EXCEPTION_IF_NULL(var_epsilon);
  auto *std = context->network()->addUnary(*var_epsilon->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
  MS_EXCEPTION_IF_NULL(std);
  auto *div =
    context->network()->addElementWise(*sub->getOutput(0), *std->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);
  MS_EXCEPTION_IF_NULL(div);
  auto *mul = context->network()->addElementWise(*div->getOutput(0), *ToTensor(&inputs[1], param_shape, context),
                                                 nvinfer1::ElementWiseOperation::kPROD);
  MS_EXCEPTION_IF_NULL(mul);
  auto *add = context->network()->addElementWise(*mul->getOutput(0), *ToTensor(&inputs[2], param_shape, context),
                                                 nvinfer1::ElementWiseOperation::kSUM);
  MS_EXCEPTION_IF_NULL(add);

  return {true, {add->getOutput(0)}};
}

MS_TRT_CONVERTER_FUNC_REG(Return) {
  std::vector<LayerInput> inputs;
  bool ret = context->LoadLayerInput(node, &inputs);
  if (!ret) {
    return {false, {}};
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    nvinfer1::ITensor *input = nullptr;
    if (inputs[i].IsTensor()) {
      input = inputs[i].tensor();
    } else {
      std::vector<size_t> shape;
      std::transform(inputs[i].shape().begin(), inputs[i].shape().end(), std::back_inserter(shape),
                     [](int64_t d) { return LongToSize(d); });
      input = ToTensor(&inputs[i], shape, context);
    }

    const std::string &name = "return_output_" + std::to_string(i);
    input->setName(name.c_str());
    context->network()->markOutput(*input);
  }

  return {true, {}};
}
}  // namespace opt
}  // namespace mindspore
