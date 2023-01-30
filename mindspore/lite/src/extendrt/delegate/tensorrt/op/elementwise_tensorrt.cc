/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/op/elementwise_tensorrt.h"
#include <unordered_map>
#include <unordered_set>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "src/extendrt/delegate/tensorrt/op/activation_tensorrt.h"
#include "ops/fusion/sub_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/fusion/reduce_fusion.h"
#include "ops/fusion/pow_fusion.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/real_div.h"
#include "ops/floor_div.h"
#include "ops/eltwise.h"
#include "ops/minimum.h"
#include "ops/maximum.h"
#include "ops/bias_add.h"
#include "ops/equal.h"
#include "ops/not_equal.h"
#include "ops/less.h"
#include "ops/greater.h"
#include "ops/floor_mod.h"

namespace mindspore::lite {
namespace {
std::unordered_map<std::string, nvinfer1::ElementWiseOperation> NOT_BOOL_PRIM2NV_ELEM_OP = {
#if TRT_VERSION_GE(7, 2)
  {ops::kNameLess, nvinfer1::ElementWiseOperation::kLESS},
  {ops::kNameGreater, nvinfer1::ElementWiseOperation::kGREATER},
#endif
  {ops::kNameAddFusion, nvinfer1::ElementWiseOperation::kSUM},
  {ops::kNamePowFusion, nvinfer1::ElementWiseOperation::kPOW},
  {ops::kNameDivFusion, nvinfer1::ElementWiseOperation::kDIV},
  {ops::kNameRealDiv, nvinfer1::ElementWiseOperation::kDIV},
  {ops::kNameFloorDiv, nvinfer1::ElementWiseOperation::kFLOOR_DIV},
  {ops::kNameSubFusion, nvinfer1::ElementWiseOperation::kSUB},
  {ops::kNameMulFusion, nvinfer1::ElementWiseOperation::kPROD},
  {ops::kNameMinimum, nvinfer1::ElementWiseOperation::kMIN},
  {ops::kNameMaximum, nvinfer1::ElementWiseOperation::kMAX},
  {ops::kNameBiasAdd, nvinfer1::ElementWiseOperation::kSUM},
#if TRT_VERSION_GE(7, 2)
  {ops::kNameEqual, nvinfer1::ElementWiseOperation::kEQUAL},
  {ops::kNameNotEqual, nvinfer1::ElementWiseOperation::kEQUAL},
#endif
};
}  // namespace

int ElementWiseTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                   const std::vector<TensorInfo> &out_tensors) {
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
                    [](const TensorInfo &tensor) { return tensor.DataType() == DataType::kNumberTypeBool; })) {
      MS_LOG(ERROR) << "invalid input type for : " << op_name_;
      return RET_ERROR;
    }
    element_wise_op_ = NOT_BOOL_PRIM2NV_ELEM_OP[type_];
  }
  if (!is_not_bool_arith) {
    // PrimitiveType_Eltwise
    auto eltwise_op = AsOps<ops::Eltwise>();
    if (eltwise_op == nullptr) {
      MS_LOG(ERROR) << "convert to Eltwise failed: " << op_name_;
      return RET_ERROR;
    }
    EltwiseMode eltwiseMode = eltwise_op->get_mode();
    std::map<EltwiseMode, nvinfer1::ElementWiseOperation> eltwise_modes = {
      {EltwiseMode::SUM, nvinfer1::ElementWiseOperation::kSUM},
      {EltwiseMode::PROD, nvinfer1::ElementWiseOperation::kPROD},
      {EltwiseMode::MAXIMUM, nvinfer1::ElementWiseOperation::kMAX},
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

void ElementWiseTensorRT::LogicalOpChangeInputType(TensorRTContext *ctx, ITensorHelper *x_input,
                                                   ITensorHelper *y_input) {
  if (type_ == ops::kNameGreater || type_ == ops::kNameLess) {
    if (x_input->trt_tensor_->getType() != nvinfer1::DataType::kINT32) {
      x_input->trt_tensor_ =
        TRTTensorCast(ctx, x_input->trt_tensor_, nvinfer1::DataType::kINT32, op_name_ + "_input_cast_to_int_0");
    }
    if (y_input->trt_tensor_->getType() != nvinfer1::DataType::kINT32) {
      y_input->trt_tensor_ =
        TRTTensorCast(ctx, y_input->trt_tensor_, nvinfer1::DataType::kINT32, op_name_ + "_input_cast_to_int_1");
    }
  }
}

int ElementWiseTensorRT::AddInnerOp(TensorRTContext *ctx) {
  ITensorHelper x_input;
  ITensorHelper y_input;
  int ret = PreprocessInputTensors(ctx, &x_input, &y_input);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PreprocessInputTensors failed.";
    return RET_ERROR;
  }
  nvinfer1::IElementWiseLayer *cal_layer;
  if (type_ == ops::kNameFloorMod) {
    cal_layer = AddFoorMod(ctx, x_input.trt_tensor_, y_input.trt_tensor_);
  } else {
    LogicalOpChangeInputType(ctx, &x_input, &y_input);
    cal_layer = ctx->network()->addElementWise(*x_input.trt_tensor_, *y_input.trt_tensor_, element_wise_op_);
  }

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
  if (type_ == ops::kNamePowFusion) {
    auto pow_op = AsOps<ops::PowFusion>();
    if (pow_op == nullptr) {
      MS_LOG(ERROR) << "PowFusion convert failed.";
      return RET_ERROR;
    }
    float scale = pow_op->get_scale();
    float shift = pow_op->get_shift();
    if (abs(scale - 1) >= 1.0e-05 || abs(shift - 0) >= 1.0e-05) {
      MS_LOG(WARNING) << "deal with scale and shift for pow op";
    }
  }
#if TRT_VERSION_GE(7, 2)
  if (type_ == ops::kNameNotEqual) {
    op_out_tensor = ctx->network()->addUnary(*op_out_tensor, nvinfer1::UnaryOperation::kNOT)->getOutput(0);
  }
  std::unordered_set<std::string> bool_producer_ops = {ops::kNameNotEqual, ops::kNameEqual, ops::kNameGreater,
                                                       ops::kNameLess};
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
  auto is_tensor = x_input.is_tensor || y_input.is_tensor;
  auto output_helper = ITensorHelper{op_out_tensor, x_input.format_, x_input.same_format_, is_tensor};
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
  if (in_tensors_[0].DataType() != in_tensors_[1].DataType()) {
    MS_LOG(INFO) << "trt op elementwise layer not support different input data type, cast to higher one";
    auto higher_index = in_tensors_[0].DataType() > in_tensors_[1].DataType() ? 0 : 1;
    auto highter_trt_tensor = input(ctx, higher_index).trt_tensor_;
    auto cast_layer = ctx->network()->addIdentity(*input(ctx, 1 - higher_index).trt_tensor_);
    CHECK_NULL_RETURN(cast_layer);
    cast_layer->setOutputType(0, highter_trt_tensor->getType());
    auto cast_output = cast_layer->getOutput(0);
    CHECK_NULL_RETURN(cast_output);
    ctx->RegisterTensor(
      ITensorHelper{cast_output, input(ctx, higher_index).format_, input(ctx, higher_index).same_format_},
      out_tensors_[0].Name());
    cast_layer->setName((op_name_ + "_cast").c_str());
    if (higher_index != 0) {
      x_input->trt_tensor_ = cast_output;
    } else {
      y_input->trt_tensor_ = cast_output;
    }
  }

  MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(*x_input);
  MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(*y_input);
  if (BroadcastInputTensors(ctx, x_input, y_input) != RET_OK) {
    return RET_ERROR;
  }

  while (x_input->trt_tensor_->getDimensions().nbDims < y_input->trt_tensor_->getDimensions().nbDims) {
    x_input->trt_tensor_ = ExpandDim(ctx, x_input->trt_tensor_, 0);
  }
  while (x_input->trt_tensor_->getDimensions().nbDims > y_input->trt_tensor_->getDimensions().nbDims) {
    y_input->trt_tensor_ = ExpandDim(ctx, y_input->trt_tensor_, 0);
  }
  return RET_OK;
}

int ElementWiseTensorRT::BroadcastInputTensors(TensorRTContext *ctx, ITensorHelper *x_input, ITensorHelper *y_input) {
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
    return RET_OK;
  } else if (GetDimsVolume(x_input->trt_tensor_->getDimensions()) !=
               GetDimsVolume(y_input->trt_tensor_->getDimensions()) &&
             x_input->trt_tensor_->getDimensions().nbDims != y_input->trt_tensor_->getDimensions().nbDims) {
    bool x_large = x_input->trt_tensor_->getDimensions().nbDims > y_input->trt_tensor_->getDimensions().nbDims;
    auto input_tensor = x_large ? y_input : x_input;
    auto output_dim = x_large ? x_input->trt_tensor_->getDimensions() : y_input->trt_tensor_->getDimensions();
    nvinfer1::Dims in_tensor_dims = input_tensor->trt_tensor_->getDimensions();
    while (in_tensor_dims.nbDims < output_dim.nbDims) {
      input_tensor->trt_tensor_ = ExpandDim(ctx, input_tensor->trt_tensor_, 0);
      in_tensor_dims = input_tensor->trt_tensor_->getDimensions();
    }
    return RET_OK;
  } else {
    return RET_OK;
  }
}

nvinfer1::IElementWiseLayer *ElementWiseTensorRT::AddFoorMod(TensorRTContext *ctx, nvinfer1::ITensor *x0_trt,
                                                             nvinfer1::ITensor *x1_trt) {
  nvinfer1::IElementWiseLayer *layer_0 =
    ctx->network()->addElementWise(*x0_trt, *x1_trt, nvinfer1::ElementWiseOperation::kFLOOR_DIV);
  layer_0->setName((op_name_ + "_floor_div").c_str());
  auto result_0 = layer_0->getOutput(0);

  nvinfer1::IElementWiseLayer *layer_1 =
    ctx->network()->addElementWise(*result_0, *x1_trt, nvinfer1::ElementWiseOperation::kPROD);
  layer_1->setName((op_name_ + "_prod").c_str());
  auto result_1 = layer_1->getOutput(0);

  nvinfer1::IElementWiseLayer *layer_2 =
    ctx->network()->addElementWise(*x0_trt, *result_1, nvinfer1::ElementWiseOperation::kSUB);
  layer_2->setName((op_name_ + "_sub").c_str());

  return layer_2;
}

nvinfer1::ITensor *ElementWiseTensorRT::AddActivation(TensorRTContext *ctx, nvinfer1::ITensor *in_tensor) {
  ActivationType activation = ActivationType::NO_ACTIVATION;
  if (type_ == ops::kNameAddFusion) {
    auto sum_op = AsOps<ops::AddFusion>();
    if (sum_op == nullptr) {
      MS_LOG(ERROR) << "AddFusion convert failed.";
      return nullptr;
    }
    if (sum_op->HasAttr(ops::kActivationType)) {
      activation = sum_op->get_activation_type();
    }
  } else if (type_ == ops::kNameDivFusion) {
    auto div_op = AsOps<ops::DivFusion>();
    if (div_op == nullptr) {
      MS_LOG(ERROR) << "DivFusion convert failed.";
      return nullptr;
    }
    if (div_op->HasAttr(ops::kActivationType)) {
      activation = div_op->get_activation_type();
    }
  } else if (type_ == ops::kNameSubFusion) {
    auto sub_op = AsOps<ops::SubFusion>();
    if (sub_op == nullptr) {
      MS_LOG(ERROR) << "SubFusion convert failed.";
      return nullptr;
    }
    if (sub_op->HasAttr(ops::kActivationType)) {
      activation = sub_op->get_activation_type();
    }
  } else if (type_ == ops::kNameMulFusion) {
    auto mul_op = AsOps<ops::MulFusion>();
    if (mul_op == nullptr) {
      MS_LOG(ERROR) << "MulFusion convert failed.";
      return nullptr;
    }
    if (mul_op->HasAttr(ops::kActivationType)) {
      activation = mul_op->get_activation_type();
    }
  } else {
    MS_LOG(DEBUG) << "no activation need for: " << op_name_;
  }
  nvinfer1::ITensor *activation_out_tensor = nullptr;
  if (activation != ActivationType::NO_ACTIVATION) {
    auto activation_layer =
      ActivationTensorRT::AddActivation(ctx, activation, 0, 0, 0, in_tensor, op_name_, device_id_);
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
  int const_tensor_index = in_tensors_[0].IsConst() ? 0 : 1;
  if (in_tensors_[0].IsConst() && in_tensors_[1].IsConst()) {
    auto large_size_index = in_tensors_[0].ElementNum() >= in_tensors_[1].ElementNum() ? 0 : 1;
    const_tensor_index = 1 - large_size_index;
  }
  auto expect_shape = ConvertMSShape(input(ctx, 1 - const_tensor_index).trt_tensor_->getDimensions());
  auto &const_tensor = in_tensors_[const_tensor_index];
  nvinfer1::ITensor *constant_input = ConvertConstantTensorWithDims(ctx, const_tensor, expect_shape, op_name_);
  CHECK_NULL_RETURN(constant_input);
  auto const_shape = const_tensor.Shape();
  auto is_scalar = const_shape.empty();
  auto const_helper = ITensorHelper{constant_input, input(ctx, 1 - const_tensor_index).format_, true, !is_scalar};
  ctx->RegisterTensor(const_helper, const_tensor.Name());
  return RET_OK;
}

REGISTER_TENSORRT_CREATOR(ops::kNameSubFusion, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameDivFusion, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameRealDiv, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNamePowFusion, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameAddFusion, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameMulFusion, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameEltwise, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameMinimum, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameMaximum, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameBiasAdd, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameFloorMod, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameFloorDiv, ElementWiseTensorRT)
#if TRT_VERSION_GE(7, 2)
REGISTER_TENSORRT_CREATOR(ops::kNameNotEqual, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameEqual, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameLess, ElementWiseTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameGreater, ElementWiseTensorRT)
#endif
}  // namespace mindspore::lite
