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

#include <unordered_map>
#include <unordered_set>
#include "src/extendrt/delegate/tensorrt/op/elementwise_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "src/extendrt/delegate/tensorrt/op/activation_tensorrt.h"

namespace mindspore::lite {
namespace {
std::unordered_map<schema::PrimitiveType, nvinfer1::ElementWiseOperation> NOT_BOOL_PRIM2NV_ELEM_OP = {
#if TRT_VERSION_GE(7, 2)
  {schema::PrimitiveType_Less, nvinfer1::ElementWiseOperation::kLESS},
  {schema::PrimitiveType_Greater, nvinfer1::ElementWiseOperation::kGREATER},
#endif
  {schema::PrimitiveType_AddFusion, nvinfer1::ElementWiseOperation::kSUM},
  {schema::PrimitiveType_PowFusion, nvinfer1::ElementWiseOperation::kPOW},
  {schema::PrimitiveType_DivFusion, nvinfer1::ElementWiseOperation::kDIV},
  {schema::PrimitiveType_RealDiv, nvinfer1::ElementWiseOperation::kDIV},
  {schema::PrimitiveType_FloorDiv, nvinfer1::ElementWiseOperation::kFLOOR_DIV},
  {schema::PrimitiveType_SubFusion, nvinfer1::ElementWiseOperation::kSUB},
  {schema::PrimitiveType_MulFusion, nvinfer1::ElementWiseOperation::kPROD},
  {schema::PrimitiveType_Minimum, nvinfer1::ElementWiseOperation::kMIN},
  {schema::PrimitiveType_Maximum, nvinfer1::ElementWiseOperation::kMAX},
  {schema::PrimitiveType_BiasAdd, nvinfer1::ElementWiseOperation::kSUM},
#if TRT_VERSION_GE(7, 2)
  {schema::PrimitiveType_Equal, nvinfer1::ElementWiseOperation::kEQUAL},
#endif
};
}  // namespace

int ElementWiseTensorRT::IsSupport(const schema::Primitive *primitive,
                                   const std::vector<mindspore::MSTensor> &in_tensors,
                                   const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "invalid input tensort size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensort size: " << out_tensors.size();
    return RET_ERROR;
  }

  bool is_not_bool_arith = NOT_BOOL_PRIM2NV_ELEM_OP.find(type_) != NOT_BOOL_PRIM2NV_ELEM_OP.end();
  if (is_not_bool_arith) {
    if (std::any_of(in_tensors.begin(), in_tensors.end(),
                    [](const mindspore::MSTensor &tensor) { return tensor.DataType() == DataType::kNumberTypeBool; })) {
      MS_LOG(ERROR) << "invalid input type for : " << op_name_;
      return RET_ERROR;
    }
    element_wise_op_ = NOT_BOOL_PRIM2NV_ELEM_OP[type_];
  }
  if (!is_not_bool_arith) {
    // PrimitiveType_Eltwise
    auto eltwise_op = op_primitive_->value_as_Eltwise();
    if (eltwise_op == nullptr) {
      MS_LOG(ERROR) << "convert to Eltwise failed: " << op_name_;
      return RET_ERROR;
    }
    schema::EltwiseMode eltwiseMode = eltwise_op->mode();
    std::map<schema::EltwiseMode, nvinfer1::ElementWiseOperation> eltwise_modes = {
      {schema::EltwiseMode::EltwiseMode_SUM, nvinfer1::ElementWiseOperation::kSUM},
      {schema::EltwiseMode::EltwiseMode_PROD, nvinfer1::ElementWiseOperation::kPROD},
      {schema::EltwiseMode::EltwiseMode_MAXIMUM, nvinfer1::ElementWiseOperation::kMAX},
    };
    auto iter_mode = eltwise_modes.find(eltwiseMode);
    if (iter_mode != eltwise_modes.end()) {
      element_wise_op_ = iter_mode->second;
    } else {
      MS_LOG(ERROR) << "unsupported type for ElementWise op" << op_name_;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ElementWiseTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network or input tensor size is invalid";
    return RET_ERROR;
  }
  ITensorHelper x_input;
  ITensorHelper y_input;
  int ret = PreprocessInputTensors(ctx, &x_input, &y_input);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PreprocessInputTensors failed.";
    return RET_ERROR;
  }
  nvinfer1::IElementWiseLayer *cal_layer =
    ctx->network()->addElementWise(*x_input.trt_tensor_, *y_input.trt_tensor_, element_wise_op_);

  if (cal_layer == nullptr) {
    MS_LOG(ERROR) << "addElementWise failed for TensorRT.";
    return RET_ERROR;
  }
  cal_layer->setName(op_name_.c_str());
  this->layer_ = cal_layer;
  ctx->RegisterLayer(cal_layer, op_name_);

  nvinfer1::ITensor *op_out_tensor = cal_layer->getOutput(0);
  if (op_out_tensor == nullptr) {
    MS_LOG(ERROR) << "addElementWise out tensor is nullptr.";
    return RET_ERROR;
  }
  // add activation
  nvinfer1::ITensor *activation_out_tensor = AddActivation(ctx, op_out_tensor);
  op_out_tensor = (activation_out_tensor == nullptr) ? op_out_tensor : activation_out_tensor;

  // scale and shift
  if (type_ == schema::PrimitiveType_PowFusion) {
    auto pow_op = op_primitive_->value_as_PowFusion();
    if (pow_op == nullptr) {
      MS_LOG(ERROR) << "PowFusion convert failed.";
      return RET_ERROR;
    }
    float scale = pow_op->scale();
    float shift = pow_op->shift();
    if (abs(scale - 1) >= 1.0e-05 || abs(shift - 0) >= 1.0e-05) {
      MS_LOG(WARNING) << "deal with scale and shift for pow op";
    }
  }
#if TRT_VERSION_GE(7, 2)
  std::unordered_set<schema::PrimitiveType> bool_producer_ops = {
    schema::PrimitiveType_Equal, schema::PrimitiveType_Greater, schema::PrimitiveType_Less};
  if (bool_producer_ops.find(type_) != bool_producer_ops.end()) {
    auto cast_layer = ctx->network()->addIdentity(*op_out_tensor);
    if (cast_layer == nullptr) {
      MS_LOG(ERROR) << "create cast layer failed for: " << op_name_;
      return RET_ERROR;
    }
    cast_layer->setOutputType(0, nvinfer1::DataType::kINT32);
    op_out_tensor = cast_layer->getOutput(0);
    MS_LOG(INFO) << "bool result cast to int32" << op_name_;
  }
#endif
  auto output_helper =
    ITensorHelper{op_out_tensor, x_input.format_, x_input.same_format_, x_input.is_tensor_ || y_input.is_tensor_};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(output_helper);
  return RET_OK;
}

int ElementWiseTensorRT::PreprocessInputTensors(TensorRTContext *ctx, ITensorHelper *x_input, ITensorHelper *y_input) {
  if (HasConst()) {
    int ret = AddConstTensor(ctx);
    if (ret != RET_OK) {
      return ret;
    }
  }
  *x_input = input(ctx, 0);
  *y_input = input(ctx, 1);
  int const_tensor_index = (in_tensors_[0].Data() != nullptr && in_tensors_[0].IsConst()) ? 0 : 1;
  auto input_helper = const_tensor_index == 0 ? y_input : x_input;
  auto const_helper = const_tensor_index == 0 ? x_input : y_input;
  MS_LOG(DEBUG) << "before transpose " << GetTensorFormat(*x_input);
  MS_LOG(DEBUG) << "before transpose " << GetTensorFormat(*y_input);

  if (input_helper->trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      input_helper->format_ != const_helper->format_) {
    // when inputs format are different, change to NHWC
    auto need_trans = input_helper->format_ == Format::NCHW ? input_helper : const_helper;
    nvinfer1::IShuffleLayer *transpose_layer = NCHW2NHWC(ctx, *need_trans->trt_tensor_);
    if (transpose_layer == nullptr) {
      MS_LOG(ERROR) << "op action convert failed";
      return RET_ERROR;
    }
    transpose_layer->setName((op_name_ + "_input_transpose2NHWC").c_str());
    need_trans->trt_tensor_ = transpose_layer->getOutput(0);
    need_trans->format_ = Format::NHWC;
    need_trans->same_format_ = true;
  }
  MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(*x_input);
  MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(*y_input);
  if (GetDimsVolume(x_input->trt_tensor_->getDimensions()) == GetDimsVolume(y_input->trt_tensor_->getDimensions()) &&
      x_input->trt_tensor_->getDimensions().nbDims != y_input->trt_tensor_->getDimensions().nbDims) {
    bool x_large = x_input->trt_tensor_->getDimensions().nbDims > y_input->trt_tensor_->getDimensions().nbDims;
    auto input_tensor = x_large ? y_input : x_input;
    auto output_dim = x_large ? x_input->trt_tensor_->getDimensions() : y_input->trt_tensor_->getDimensions();
    auto reshape_layer = ctx->network()->addShuffle(*input_tensor->trt_tensor_);
    if (reshape_layer == nullptr) {
      MS_LOG(ERROR) << "add reshape failed for " << op_name_;
      return RET_ERROR;
    }
    reshape_layer->setReshapeDimensions(output_dim);
    input_tensor->trt_tensor_ = reshape_layer->getOutput(0);
  }
  return RET_OK;
}

nvinfer1::ITensor *ElementWiseTensorRT::AddActivation(TensorRTContext *ctx, nvinfer1::ITensor *in_tensor) {
  schema::ActivationType activation = schema::ActivationType::ActivationType_NO_ACTIVATION;
  switch (type_) {
    case schema::PrimitiveType_AddFusion: {
      auto sum_op = op_primitive_->value_as_AddFusion();
      if (sum_op == nullptr) {
        MS_LOG(ERROR) << "AddFusion convert failed.";
        return nullptr;
      }
      activation = sum_op->activation_type();
      break;
    }
    case schema::PrimitiveType_DivFusion: {
      auto div_op = op_primitive_->value_as_DivFusion();
      if (div_op == nullptr) {
        MS_LOG(ERROR) << "DivFusion convert failed.";
        return nullptr;
      }
      activation = div_op->activation_type();
      break;
    }
    case schema::PrimitiveType_SubFusion: {
      auto sub_op = op_primitive_->value_as_SubFusion();
      if (sub_op == nullptr) {
        MS_LOG(ERROR) << "SubFusion convert failed.";
        return nullptr;
      }
      activation = sub_op->activation_type();
      break;
    }
    case schema::PrimitiveType_MulFusion: {
      auto mul_op = op_primitive_->value_as_MulFusion();
      if (mul_op == nullptr) {
        MS_LOG(ERROR) << "MulFusion convert failed.";
        return nullptr;
      }
      activation = mul_op->activation_type();
      break;
    }
    default:
      MS_LOG(DEBUG) << "no activation need for: " << op_name_;
  }
  nvinfer1::ITensor *activation_out_tensor = nullptr;
  if (activation != schema::ActivationType::ActivationType_NO_ACTIVATION) {
    auto activation_layer = ActivationTensorRT::AddActivation(ctx, activation, 0, 0, 0, in_tensor, device_id_);
    if (activation_layer == nullptr) {
      MS_LOG(ERROR) << "addActivation for element wise failed";
      return nullptr;
    }
    activation_layer->setName((op_name_ + "_activation").c_str());
    activation_out_tensor = activation_layer->getOutput(0);
  }
  return activation_out_tensor;
}

int ElementWiseTensorRT::AddConstTensor(TensorRTContext *ctx) {
  int const_tensor_index = (in_tensors_[0].Data() != nullptr && in_tensors_[0].IsConst()) ? 0 : 1;
  auto expect_shape = ConvertMSShape(input(ctx, 1 - const_tensor_index).trt_tensor_->getDimensions());
  nvinfer1::ITensor *constant_input =
    ConvertConstantTensorWithDims(ctx, in_tensors_[const_tensor_index], expect_shape, op_name_);
  CHECK_NULL_RETURN(constant_input);
  bool is_tensor = !in_tensors_[const_tensor_index].Shape().empty();
  auto const_helper = ITensorHelper{constant_input, input(ctx, 1 - const_tensor_index).format_,
                                    input(ctx, 1 - const_tensor_index).same_format_, is_tensor};
  ctx->RegisterTensor(const_helper, in_tensors_[const_tensor_index].Name());
  return RET_OK;
}

REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_SubFusion, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_DivFusion, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_RealDiv, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_PowFusion, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_AddFusion, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_MulFusion, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Eltwise, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Minimum, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Maximum, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_BiasAdd, ElementWiseTensorRT)
#if TRT_VERSION_GE(7, 2)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Equal, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Less, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Greater, ElementWiseTensorRT)
#endif
}  // namespace mindspore::lite
