/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/populate_parameter.h"
#include <float.h>
#include "src/ops/ops.h"
#include "utils/log_adapter.h"
#include "src/runtime/kernel/arm/opclib/op_base.h"
#include "src/runtime/kernel/arm/opclib/fp32/arg_min_max.h"
#include "src/runtime/kernel/arm/opclib/fp32/cast.h"
#include "src/runtime/kernel/arm/opclib/concat_parameter.h"
#include "src/runtime/kernel/arm/opclib/fp32/slice.h"
#include "src/runtime/kernel/arm/opclib/fp32/broadcast_to.h"
#include "src/runtime/kernel/arm/opclib/reshape_parameter.h"
#include "src/runtime/kernel/arm/opclib/fp32/stack.h"
#include "src/runtime/kernel/arm/opclib/unstack.h"
#include "src/runtime/kernel/arm/opclib/fp32/depth_to_space.h"
#include "src/runtime/kernel/arm/opclib/conv_parameter.h"
#include "src/runtime/kernel/arm/opclib/fp32/pooling.h"
#include "src/runtime/kernel/arm/opclib/matmul.h"
#include "src/runtime/kernel/arm/opclib/fp32/softmax.h"
#include "src/runtime/kernel/arm/opclib/tile.h"
#include "src/runtime/kernel/arm/opclib/topk.h"
#include "src/runtime/kernel/arm/opclib/fp32/reduce.h"
#include "src/runtime/kernel/arm/opclib/fp32/activation.h"
#include "src/runtime/kernel/arm/opclib/fp32/arithmetic.h"
#include "src/runtime/kernel/arm/opclib/fused_batchnorm.h"
#include "src/runtime/kernel/arm/opclib/power.h"
#include "src/runtime/kernel/arm/opclib/fp32/range.h"
#include "src/runtime/kernel/arm/opclib/fp32/local_response_norm.h"
#include "src/runtime/kernel/arm/opclib/fp32/expandDims.h"
#include "src/runtime/kernel/arm/opclib/fp32/arithmetic_self.h"
#include "src/runtime/kernel/arm/opclib/pad_parameter.h"
#include "src/runtime/kernel/arm/opclib/fp32/fill.h"
#include "src/runtime/kernel/arm/opclib/transpose.h"
#include "src/runtime/kernel/arm/opclib/split.h"
#include "src/runtime/kernel/arm/opclib/squeeze.h"
#include "src/runtime/kernel/arm/opclib/fp32/gather.h"
#include "src/runtime/kernel/arm/opclib/fp32/reverse.h"
#include "src/runtime/kernel/arm/opclib/reverse_sequence.h"
#include "src/runtime/kernel/arm/opclib/unique.h"
#include "src/runtime/kernel/arm/opclib/scale.h"
#include "src/runtime/kernel/arm/opclib/fp32/gatherNd.h"
#include "src/runtime/kernel/arm/opclib/resize.h"
#include "src/runtime/kernel/arm/opclib/scatter_nd.h"
#include "src/runtime/kernel/arm/opclib/fp32/batch_to_space.h"
#include "src/runtime/kernel/arm/opclib/fp32/crop.h"
#include "src/runtime/kernel/arm/fp32/flatten.h"
#include "src/runtime/kernel/arm/opclib/fp32/unsqueeze.h"
#include "src/runtime/kernel/arm/opclib/fp32/one_hot.h"
#include "src/runtime/kernel/arm/opclib/fp32/strided_slice.h"

namespace mindspore::kernel {
FillParameter *PopulateFillParam(const lite::Primitive *primitive) {
  auto param = primitive->Value()->value_as_Fill();
  FillParameter *parameter = new (std::nothrow) FillParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new FillParameter failed.";
    return nullptr;
  }
  auto flatDims = param->dims();
  parameter->num_dims_ = flatDims->size();
  int i = 0;
  for (auto iter = flatDims->begin(); iter != flatDims->end(); iter++) {
    parameter->dims_[i++] = *iter;
  }
  return parameter;
}

ExpandDimsParameter *PopulateExpandDimsParam(const lite::Primitive *primitive) {
  auto param = primitive->Value()->value_as_ExpandDims();
  ExpandDimsParameter *parameter = new (std::nothrow) ExpandDimsParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ExpandDimsParameter failed.";
    return nullptr;
  }
  parameter->dim_ = param->dim();
  return parameter;
}

PoolingParameter *PopulatePoolingParam(const lite::Primitive *primitive) {
  auto pooling_primitive = primitive->Value()->value_as_Pooling();
  // todo use malloc instead
  PoolingParameter *parameter = new (std::nothrow) PoolingParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new PoolingParameter failed.";
    return nullptr;
  }
  parameter->global_ = pooling_primitive->global();
  parameter->window_w_ = pooling_primitive->windowW();
  parameter->window_h_ = pooling_primitive->windowH();
  // todo format
  auto pooling_lite_primitive = (lite::Pooling *)primitive;
  MS_ASSERT(nullptr != pooling_lite_primitive);
  parameter->pad_u_ = pooling_lite_primitive->PadUp();
  parameter->pad_d_ = pooling_lite_primitive->PadDown();
  parameter->pad_l_ = pooling_lite_primitive->PadLeft();
  parameter->pad_r_ = pooling_lite_primitive->PadRight();
  parameter->stride_w_ = pooling_primitive->strideW();
  parameter->stride_h_ = pooling_primitive->strideH();

  auto pool_mode = pooling_primitive->poolingMode();
  switch (pool_mode) {
    case schema::PoolMode_MAX_POOLING:
      parameter->max_pooling_ = true;
      parameter->avg_pooling_ = false;
      break;
    case schema::PoolMode_MEAN_POOLING:
      parameter->max_pooling_ = false;
      parameter->avg_pooling_ = true;
      break;
    default:
      parameter->max_pooling_ = false;
      parameter->avg_pooling_ = false;
      break;
  }

  auto round_mode = pooling_primitive->roundMode();
  switch (round_mode) {
    case schema::RoundMode_FLOOR:
      parameter->round_floor_ = true;
      parameter->round_ceil_ = false;
      break;
    case schema::RoundMode_CEIL:
      parameter->round_floor_ = false;
      parameter->round_ceil_ = true;
      break;
    default:
      parameter->round_floor_ = false;
      parameter->round_ceil_ = false;
      break;
  }
  return parameter;
}

MatMulParameter *PopulateFullconnectionParameter(const lite::Primitive *primitive) {
  auto param = primitive->Value()->value_as_FullConnection();
  MatMulParameter *parameter = new (std::nothrow) MatMulParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new FullconnectionParameter failed.";
    return nullptr;
  }
  parameter->b_transpose_ = true;
  parameter->a_transpose_ = false;
  parameter->has_bias_ = param->hasBias();
  parameter->minf_ = -FLT_MAX;
  parameter->maxf_ = FLT_MAX;
  return parameter;
}

MatMulParameter *PopulateMatMulParameter(const lite::Primitive *primitive) {
  auto param = primitive->Value()->value_as_MatMul();
  MatMulParameter *parameter = new (std::nothrow) MatMulParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new FullconnectionParameter failed.";
    return nullptr;
  }
  parameter->b_transpose_ = param->transposeB();
  parameter->a_transpose_ = param->transposeA();
  parameter->has_bias_ = false;
  parameter->minf_ = -FLT_MAX;
  parameter->maxf_ = FLT_MAX;
  return parameter;
}

ConvParameter *PopulateConvParameter(const lite::Primitive *primitive) {
  ConvParameter *parameter = new (std::nothrow) ConvParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }
  auto conv_primitive = primitive->Value()->value_as_Conv2D();
  parameter->kernel_h_ = conv_primitive->kernelH();
  parameter->kernel_w_ = conv_primitive->kernelW();
  // todo format
  parameter->group_ = conv_primitive->group();
  parameter->stride_h_ = conv_primitive->strideH();
  parameter->stride_w_ = conv_primitive->strideW();

  auto conv2d_lite_primitive = (lite::Conv2D *)primitive;
  MS_ASSERT(nullptr != conv2d_lite_primitive);
  parameter->pad_u_ = conv2d_lite_primitive->PadUp();
  parameter->pad_d_ = conv2d_lite_primitive->PadDown();
  parameter->pad_l_ = conv2d_lite_primitive->PadLeft();
  parameter->pad_r_ = conv2d_lite_primitive->PadRight();
  parameter->pad_h_ = conv2d_lite_primitive->PadUp();
  parameter->pad_w_ = conv2d_lite_primitive->PadLeft();
  parameter->dilation_h_ = conv_primitive->dilateH();
  parameter->dilation_w_ = conv_primitive->dilateW();
  parameter->input_channel_ = conv_primitive->channelIn();
  parameter->output_channel_ = conv_primitive->channelOut();
  parameter->group_ = conv_primitive->group();
  auto act_type = conv_primitive->activationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      parameter->is_relu_ = true;
      parameter->is_relu6_ = false;
      break;
    case schema::ActivationType_RELU6:
      parameter->is_relu_ = false;
      parameter->is_relu6_ = true;
      break;
    default:
      parameter->is_relu_ = false;
      parameter->is_relu6_ = false;
      break;
  }
  return parameter;
}

ConvParameter *PopulateConvDwParameter(const lite::Primitive *primitive) {
  ConvParameter *parameter = new (std::nothrow) ConvParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }
  auto conv_primitive = primitive->Value()->value_as_DepthwiseConv2D();
  parameter->kernel_h_ = conv_primitive->kernelH();
  parameter->kernel_w_ = conv_primitive->kernelW();
  // todo format, group
  parameter->stride_h_ = conv_primitive->strideH();
  parameter->stride_w_ = conv_primitive->strideW();

  auto pad_mode = conv_primitive->padMode();
  auto convdw_lite_primitive = (lite::DepthwiseConv2D *)primitive;
  MS_ASSERT(nullptr != convdw_lite_primitive);
  parameter->pad_u_ = convdw_lite_primitive->PadUp();
  parameter->pad_d_ = convdw_lite_primitive->PadDown();
  parameter->pad_l_ = convdw_lite_primitive->PadLeft();
  parameter->pad_r_ = convdw_lite_primitive->PadRight();
  parameter->pad_h_ = convdw_lite_primitive->PadUp();
  parameter->pad_w_ = convdw_lite_primitive->PadLeft();
  parameter->dilation_h_ = conv_primitive->dilateH();
  parameter->dilation_w_ = conv_primitive->dilateW();
  auto act_type = conv_primitive->activationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      parameter->is_relu_ = true;
      parameter->is_relu6_ = false;
      break;
    case schema::ActivationType_RELU6:
      parameter->is_relu_ = false;
      parameter->is_relu6_ = true;
      break;
    default:
      parameter->is_relu_ = false;
      parameter->is_relu6_ = false;
      break;
  }
  return parameter;
}

ConvParameter *PopulateDeconvDwParameter(const lite::Primitive *primitive) {
  ConvParameter *parameter = new ConvParameter();
  auto conv_primitive = primitive->Value()->value_as_DeDepthwiseConv2D();
  parameter->kernel_h_ = conv_primitive->kernelH();
  parameter->kernel_w_ = conv_primitive->kernelW();
  // todo format, group
  parameter->stride_h_ = conv_primitive->strideH();
  parameter->stride_w_ = conv_primitive->strideW();

  auto deconvdw_lite_primitive = (lite::DeconvDepthwiseConv2D *)primitive;
  MS_ASSERT(nullptr != deconvdw_lite_primitive);
  parameter->pad_u_ = deconvdw_lite_primitive->PadUp();
  parameter->pad_d_ = deconvdw_lite_primitive->PadDown();
  parameter->pad_l_ = deconvdw_lite_primitive->PadLeft();
  parameter->pad_r_ = deconvdw_lite_primitive->PadRight();
  parameter->pad_h_ = deconvdw_lite_primitive->PadUp();
  parameter->pad_w_ = deconvdw_lite_primitive->PadLeft();
  parameter->dilation_h_ = conv_primitive->dilateH();
  parameter->dilation_w_ = conv_primitive->dilateW();
  auto act_type = conv_primitive->activationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      parameter->is_relu_ = true;
      parameter->is_relu6_ = false;
      break;
    case schema::ActivationType_RELU6:
      parameter->is_relu_ = false;
      parameter->is_relu6_ = true;
      break;
    default:
      parameter->is_relu_ = false;
      parameter->is_relu6_ = false;
      break;
  }
  return parameter;
}

ConvParameter *PopulateDeconvParameter(const lite::Primitive *primitive) {
  ConvParameter *parameter = new ConvParameter();
  auto conv_primitive = primitive->Value()->value_as_DeConv2D();
  parameter->kernel_h_ = conv_primitive->kernelH();
  parameter->kernel_w_ = conv_primitive->kernelW();
  parameter->stride_h_ = conv_primitive->strideH();
  parameter->stride_w_ = conv_primitive->strideW();

  auto deconv_lite_primitive = (lite::DeConv2D *)primitive;
  MS_ASSERT(nullptr != deconvdw_lite_primitive);
  parameter->pad_u_ = deconv_lite_primitive->PadUp();
  parameter->pad_d_ = deconv_lite_primitive->PadDown();
  parameter->pad_l_ = deconv_lite_primitive->PadLeft();
  parameter->pad_r_ = deconv_lite_primitive->PadRight();
  parameter->pad_h_ = deconv_lite_primitive->PadUp();
  parameter->pad_w_ = deconv_lite_primitive->PadLeft();
  parameter->dilation_h_ = conv_primitive->dilateH();
  parameter->dilation_w_ = conv_primitive->dilateW();
  auto act_type = conv_primitive->activationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      parameter->is_relu_ = true;
      parameter->is_relu6_ = false;
      break;
    case schema::ActivationType_RELU6:
      parameter->is_relu_ = false;
      parameter->is_relu6_ = true;
      break;
    default:
      parameter->is_relu_ = false;
      parameter->is_relu6_ = false;
      break;
  }
  return parameter;
}

SoftmaxParameter *PopulateSoftmaxParameter(const lite::Primitive *primitive) {
  auto softmax_primitive = primitive->Value()->value_as_SoftMax();
  SoftmaxParameter *parameter = new (std::nothrow) SoftmaxParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new SoftmaxParameter failed.";
    return nullptr;
  }
  parameter->axis_ = softmax_primitive->axis();
  return parameter;
}

ReduceParameter *PopulateReduceParameter(const lite::Primitive *primitive) {
  ReduceParameter *parameter = new (std::nothrow) ReduceParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ReduceParameter failed.";
    return nullptr;
  }
  auto reduce = primitive->Value()->value_as_Reduce();
  parameter->keep_dims_ = reduce->keepDims();
  auto axisVector = reduce->axes();
  if (axisVector->size() > REDUCE_MAX_AXES_NUM) {
    MS_LOG(ERROR) << "Reduce axes size " << axisVector->size() << " exceed limit " << REDUCE_MAX_AXES_NUM;
    delete (parameter);
    return nullptr;
  }
  parameter->num_axes_ = static_cast<int>(axisVector->size());
  int i = 0;
  for (auto iter = axisVector->begin(); iter != axisVector->end(); iter++) {
    parameter->axes_[i++] = *iter;
  }
  parameter->mode_ = static_cast<int>(reduce->mode());
  return parameter;
}

PadParameter *PopulatePadParameter(const lite::Primitive *primitive) {
  PadParameter *pad_param = new (std::nothrow) PadParameter();
  if (pad_param == nullptr) {
    MS_LOG(ERROR) << "new PadParameter failed.";
    return nullptr;
  }
  auto pad_node = primitive->Value()->value_as_Pad();

  pad_param->pad_mode_ = pad_node->paddingMode();
  if (pad_param->pad_mode_ == schema::PaddingMode_CONSTANT) {
    pad_param->constant_value_ = pad_node->constantValue();
  } else {
    MS_LOG(ERROR) << "Invalid padding mode: " << pad_param->pad_mode_;
    return nullptr;
  }

  auto size = pad_node->paddings()->size();
  if (size > MAX_PAD_SIZE) {
    MS_LOG(ERROR) << "Invalid padding size: " << size;
    return nullptr;
  }

  for (size_t i = 0; i < size; i++) {
    pad_param->paddings_[MAX_PAD_SIZE - size + i] = (*(pad_node->paddings()))[i];
  }
  return pad_param;
}

ActivationParameter *PopulateActivationParameter(const lite::Primitive *primitive) {
  ActivationParameter *parameter = new (std::nothrow) ActivationParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ActivationParameter failed.";
    return nullptr;
  }
  auto activation = primitive->Value()->value_as_Activation();
  parameter->type_ = static_cast<int>(activation->type());
  return parameter;
}

FusedBatchNormParameter *PopulateFusedBatchNorm(const lite::Primitive *primitive) {
  FusedBatchNormParameter *parameter = new (std::nothrow) FusedBatchNormParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new FusedBatchNormParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_FusedBatchNorm();
  parameter->epsilon_ = param->epsilon();
  return parameter;
}

ArithmeticParameter *PopulateArithmetic(const lite::Primitive *primitive) {
  ArithmeticParameter *parameter = new (std::nothrow) ArithmeticParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticParameter failed.";
    return nullptr;
  }
  parameter->op_parameter.type_ = primitive->Type();
  parameter->broadcasting_ = ((lite::Arithmetic *)primitive)->Broadcasting();
  parameter->ndim_ = ((lite::Arithmetic *)primitive)->NDims();
  auto tmp_shape = ((lite::Arithmetic *)primitive)->InShape0();
  (void)memcpy(parameter->in_shape0_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  tmp_shape = ((lite::Arithmetic *)primitive)->InShape1();
  (void)memcpy(parameter->in_shape1_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  tmp_shape = ((lite::Arithmetic *)primitive)->OutputShape();
  (void)memcpy(parameter->out_shape_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  return parameter;
}

ArithmeticParameter *PopulateEltwiseParam(const lite::Primitive *primitive) {
  ArithmeticParameter *parameter = new (std::nothrow) ArithmeticParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticParameter failed.";
    return nullptr;
  }
  auto eltwise = primitive->Value()->value_as_Eltwise();
  switch (eltwise->mode()) {
    case schema::EltwiseMode_PROD:
      parameter->op_parameter.type_ = schema::PrimitiveType_Mul;
      break;
    case schema::EltwiseMode_SUM:
      parameter->op_parameter.type_ = schema::PrimitiveType_Add;
      break;
    case schema::EltwiseMode_MAXIMUM:
      parameter->op_parameter.type_ = schema::PrimitiveType_Maximum;
      break;
    default:
      delete parameter;
      return nullptr;
  }
  return parameter;
}

ArithmeticSelfParameter *PopulateArithmeticSelf(const lite::Primitive *primitive) {
  ArithmeticSelfParameter *parameter = new (std::nothrow) ArithmeticSelfParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticParameter failed.";
    return nullptr;
  }
  parameter->op_parameter_.type_ = primitive->Type();
  return parameter;
}

PowerParameter *PopulatePowerParameter(const lite::Primitive *primitive) {
  PowerParameter *parameter = new (std::nothrow) PowerParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new PowerParameter failed.";
    return nullptr;
  }
  auto power = primitive->Value()->value_as_Power();
  parameter->power_ = power->power();
  parameter->scale_ = power->scale();
  parameter->shift_ = power->shift();
  return parameter;
}

ArgMinMaxParameter *PopulateArgMaxParam(const lite::Primitive *primitive) {
  ArgMinMaxParameter *parameter = new (std::nothrow) ArgMinMaxParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ArgMinMaxParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_ArgMax();
  parameter->op_parameter_.type_ = primitive->Type();
  parameter->axis_ = param->axis();
  parameter->topk_ = param->topK();
  parameter->axis_type_ = param->axisType();
  parameter->out_value_ = param->outMaxValue();
  parameter->keep_dims_ = param->keepDims();
  return parameter;
}

ArgMinMaxParameter *PopulateArgMinParam(const lite::Primitive *primitive) {
  ArgMinMaxParameter *parameter = new (std::nothrow) ArgMinMaxParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ArgMinMaxParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_ArgMin();
  parameter->op_parameter_.type_ = primitive->Type();
  parameter->axis_ = param->axis();
  parameter->topk_ = param->topK();
  parameter->axis_type_ = param->axisType();
  parameter->out_value_ = param->outMaxValue();
  parameter->keep_dims_ = param->keepDims();
  return parameter;
}

CastParameter *PopulateCastParam(const lite::Primitive *primitive) {
  CastParameter *parameter = new (std::nothrow) CastParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new CastParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_Cast();
  parameter->op_parameter_.type_ = primitive->Type();
  parameter->src_type_ = param->srcT();
  parameter->dst_type_ = param->dstT();
  return parameter;
}

LocalResponseNormParameter *PopulateLocalResponseNormParameter(const lite::Primitive *primitive) {
  auto local_response_norm_attr = primitive->Value()->value_as_LocalResponseNormalization();
  LocalResponseNormParameter *parameter = new (std::nothrow) LocalResponseNormParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new LocalResponseNormParameter failed.";
    return nullptr;
  }
  parameter->depth_radius_ = local_response_norm_attr->depth_radius();
  parameter->bias_ = local_response_norm_attr->bias();
  parameter->alpha_ = local_response_norm_attr->alpha();
  parameter->beta_ = local_response_norm_attr->beta();
  return parameter;
}

RangeParameter *PopulateRangeParameter(const lite::Primitive *primitive) {
  auto range_attr = primitive->Value()->value_as_Range();
  RangeParameter *parameter = new (std::nothrow) RangeParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new RangeParameter failed.";
    return nullptr;
  }
  parameter->start_ = range_attr->start();
  parameter->limit_ = range_attr->limit();
  parameter->delta_ = range_attr->delta();
  parameter->dType_ = range_attr->dType();
  return parameter;
}

OpParameter *PopulateCeilParameter(const lite::Primitive *primitive) {
  OpParameter *parameter = new (std::nothrow) OpParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new OpParameter failed.";
    return nullptr;
  }
  parameter->type_ = primitive->Type();
  return parameter;
}

ConcatParameter *PopulateConcatParameter(const lite::Primitive *primitive) {
  ConcatParameter *parameter = new (std::nothrow) ConcatParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ConcatParameter failed.";
    return nullptr;
  }
  parameter->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_Concat();
  parameter->axis_ = param->axis();
  return parameter;
}

TileParameter *PopulateTileParameter(const lite::Primitive *primitive) {
  TileParameter *parameter = new (std::nothrow) TileParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new TileParameter failed.";
    return nullptr;
  }
  parameter->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_Tile();
  auto multiples = param->multiples();
  parameter->in_dim_ = multiples->size();
  for (size_t i = 0; i < parameter->in_dim_; ++i) {
    parameter->multiples_[i] = multiples->Get(i);
  }
  return parameter;
}

TopkParameter *PopulateTopKParameter(const lite::Primitive *primitive) {
  TopkParameter *parameter = new (std::nothrow) TopkParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new TopkParameter failed.";
    return nullptr;
  }
  parameter->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_TopK();
  parameter->k_ = param->k();
  parameter->sorted_ = param->sorted();
  return parameter;
}

OpParameter *PopulateNhwc2NchwParameter(const lite::Primitive *primitive) {
  OpParameter *parameter = new (std::nothrow) OpParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new Nhwc2NchwParameter failed.";
    return nullptr;
  }
  parameter->type_ = primitive->Type();
  return parameter;
}

OpParameter *PopulateNchw2NhwcParameter(const lite::Primitive *primitive) {
  OpParameter *parameter = new (std::nothrow) OpParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new Nchw2NhwcParameter failed.";
    return nullptr;
  }
  parameter->type_ = primitive->Type();
  return parameter;
}

TransposeParameter *PopulateTransposeParameter(const lite::Primitive *primitive) {
  TransposeParameter *parameter = new (std::nothrow) TransposeParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new TransposeParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_Transpose();
  parameter->op_parameter_.type_ = primitive->Type();
  auto perm_vector_ = param->perm();
  int i = 0;
  for (auto iter = perm_vector_->begin(); iter != perm_vector_->end(); iter++) {
    parameter->perm_[i++] = *iter;
  }
  parameter->num_axes_ = i;
  parameter->conjugate_ = param->conjugate();
  return parameter;
}

SplitParameter *PopulateSplitParameter(const lite::Primitive *primitive) {
  SplitParameter *parameter = new (std::nothrow) SplitParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new SplitParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_Split();
  parameter->op_parameter_.type_ = primitive->Type();
  parameter->num_split_ = param->numberSplit();
  auto split_sizes_vector_ = param->sizeSplits();
  int i = 0;
  for (auto iter = split_sizes_vector_->begin(); iter != split_sizes_vector_->end(); iter++) {
    parameter->split_sizes_[i++] = *iter;
  }
  parameter->split_dim_ = param->splitDim();
  parameter->num_split_ = param->numberSplit();
  return parameter;
}

SqueezeParameter *PopulateSqueezeParameter(const lite::Primitive *primitive) {
  SqueezeParameter *parameter = new (std::nothrow) SqueezeParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new SqueezeParameter failed.";
    return nullptr;
  }
  parameter->op_parameter_.type_ = primitive->Type();
  return parameter;
}

ScaleParameter *PopulateScaleParameter(const lite::Primitive *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "input primitive is nullptr";
    return nullptr;
  }
  ScaleParameter *parameter = new (std::nothrow) ScaleParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ScaleParameter failed.";
    return nullptr;
  }
  parameter->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_Scale();
  if (param == nullptr) {
    MS_LOG(ERROR) << "value_as_Scale return nullptr";
    return nullptr;
  }
  // NCHW todo use enum
  if (param->format() == schema::Format_NCHW) {
    parameter->axis_ = 1;
    parameter->num_axis_ = 1;
  } else if (param->format() == schema::Format_NHWC) {
    parameter->axis_ = 3;
    parameter->num_axis_ = 1;
  }

  return parameter;
}

GatherParameter *PopulateGatherParameter(const lite::Primitive *primitive) {
  auto gather_attr = primitive->Value()->value_as_Gather();
  GatherParameter *parameter = new (std::nothrow) GatherParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new GatherParameter failed.";
    return nullptr;
  }
  parameter->axis_ = gather_attr->axis();
  parameter->batchDims_ = gather_attr->batchDims();
  return parameter;
}

GatherNdParameter *PopulateGatherNdParameter(const lite::Primitive *primitive) {
  GatherNdParameter *parameter = new (std::nothrow) GatherNdParameter();
  MS_ASSERT(paramter != nullptr);
  auto gatherNd_attr = primitive->Value()->value_as_GatherNd();
  parameter->batchDims_ = gatherNd_attr->batchDims();
  return parameter;
}

ScatterNDParameter *PopulateScatterNDParameter(const lite::Primitive *primitive) {
  ScatterNDParameter *parameter = new (std::nothrow) ScatterNDParameter();
  MS_ASSERT(paramter != nullptr);
  return parameter;
}

SliceParameter *PopulateSliceParam(const lite::Primitive *primitive) {
  SliceParameter *parameter = new (std::nothrow) SliceParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new SliceParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_Slice();
  parameter->op_parameter_.type_ = primitive->Type();
  auto param_begin = param->begin();
  auto param_size = param->size();
  if (param_begin->size() != param_size->size()) {
    delete parameter;
    return nullptr;
  }
  parameter->param_length_ = static_cast<int32_t>(param_begin->size());
  for (int32_t i = 0; i < parameter->param_length_; ++i) {
    parameter->begin_[i] = param_begin->Get(i);
    parameter->size_[i] = param_size->Get(i);
  }
  return parameter;
}

BroadcastToParameter *PopulateBroadcastToParam(const lite::Primitive *primitive) {
  BroadcastToParameter *parameter = new (std::nothrow) BroadcastToParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new BroadcastToParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_BroadcastTo();
  parameter->op_parameter_.type_ = primitive->Type();
  auto dst_shape = param->dst_shape();
  parameter->shape_size_ = dst_shape->size();
  for (size_t i = 0; i < parameter->shape_size_; ++i) {
    parameter->shape_[i] = dst_shape->Get(i);
  }
  return parameter;
}

ReshapeParameter *PopulateReshapeParam(const lite::Primitive *primitive) {
  ReshapeParameter *parameter = new (std::nothrow) ReshapeParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ReshapeParameter failed.";
    return nullptr;
  }
  parameter->op_parameter_.type_ = primitive->Type();
  return parameter;
}

ReverseParameter *PopulateReverseParameter(const lite::Primitive *primitive) {
  auto reverse_attr = primitive->Value()->value_as_Reverse();
  ReverseParameter *parameter = new (std::nothrow) ReverseParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ReverseParameter failed.";
    return nullptr;
  }
  auto flatAxis = reverse_attr->axis();
  parameter->num_axis_ = flatAxis->size();
  int i = 0;
  for (auto iter = flatAxis->begin(); iter != flatAxis->end(); iter++) {
    parameter->axis_[i++] = *iter;
  }
  return parameter;
}

UnsqueezeParameter *PopulateUnsqueezeParameter(const lite::Primitive *primitive) {
  auto unsqueeze_attr = primitive->Value()->value_as_Unsqueeze();
  UnsqueezeParameter *parameter = new (std::nothrow) UnsqueezeParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ReverseParameter failed.";
    return nullptr;
  }
  auto flatAxis = unsqueeze_attr->axis();
  parameter->num_dim_ = flatAxis->size();
  int i = 0;
  for (auto iter = flatAxis->begin(); iter != flatAxis->end(); iter++) {
    parameter->dims_[i++] = *iter;
  }
  return parameter;
}

StackParameter *PopulateStackParam(const lite::Primitive *primitive) {
  StackParameter *parameter = new (std::nothrow) StackParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new StackParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_Stack();
  parameter->op_parameter_.type_ = primitive->Type();
  parameter->axis_ = param->axis();
  return parameter;
}

UnstackParameter *PopulateUnstackParam(const lite::Primitive *primitive) {
  UnstackParameter *parameter = new (std::nothrow) UnstackParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new UnstackParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_Unstack();
  parameter->op_parameter_.type_ = primitive->Type();
  parameter->num_ = param->num();
  parameter->axis_ = param->axis();
  return parameter;
}

ReverseSequenceParameter *PopulateReverseSequenceParam(const lite::Primitive *primitive) {
  ReverseSequenceParameter *parameter = new (std::nothrow) ReverseSequenceParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ReverseSequenceParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_ReverseSequence();
  parameter->op_parameter_.type_ = primitive->Type();
  parameter->seq_axis_ = param->seqAxis();
  parameter->batch_axis_ = param->batchAxis();
  return parameter;
}

UniqueParameter *PopulateUniqueParam(const lite::Primitive *primitive) {
  UniqueParameter *parameter = new (std::nothrow) UniqueParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new PopulateUniqueParam failed.";
    return nullptr;
  }
  parameter->op_parameter_.type_ = primitive->Type();
  return parameter;
}

DepthToSpaceParameter *PopulateDepthToSpaceParam(const lite::Primitive *primitive) {
  DepthToSpaceParameter *parameter = new (std::nothrow) DepthToSpaceParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new DepthToSpaceParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_DepthToSpace();
  parameter->op_parameter_.type_ = primitive->Type();
  parameter->block_size_ = param->blockSize();
  return parameter;
}

ResizeParameter *PopulateResizeParameter(const lite::Primitive *primitive) {
  ResizeParameter *parameter = new (std::nothrow) ResizeParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new ResizeParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_Resize();
  parameter->method_ = param->method();
  parameter->new_height_ = param->newHeight();
  parameter->new_width_ = param->newWidth();
  parameter->align_corners_ = param->alignCorners();
  parameter->preserve_aspect_ratio_ = param->preserveAspectRatio();
  return parameter;
}

BatchToSpaceParameter *PopulateBatchToSpaceParameter(const lite::Primitive *primitive) {
  BatchToSpaceParameter *parameter = new (std::nothrow) BatchToSpaceParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "New BatchToSpaceParameter fail!";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_BatchToSpace();
  auto block_shape = param->blockShape();
  if (block_shape->size() != BATCH_TO_SPACE_BLOCK_SHAPE_SIZE) {
    MS_LOG(ERROR) << "batch_to_space blockShape size should be " << BATCH_TO_SPACE_BLOCK_SHAPE_SIZE;
    return nullptr;
  }

  auto crops = param->crops();
  if (crops->size() != BATCH_TO_SPACE_CROPS_SIZE) {
    MS_LOG(ERROR) << "batch_to_space crops size should be " << BATCH_TO_SPACE_CROPS_SIZE;
    return nullptr;
  }

  for (int i = 0; i < BATCH_TO_SPACE_BLOCK_SHAPE_SIZE; ++i) {
    parameter->block_shape_[i] = block_shape->Get(i);
  }

  for (int i = 0; i < BATCH_TO_SPACE_CROPS_SIZE; ++i) {
    parameter->crops_[i] = crops->Get(i);
  }
  return parameter;
}

CropParameter *PopulateCropParameter(const lite::Primitive *primitive) {
  auto param = primitive->Value()->value_as_Crop();
  auto param_offset = param->offsets();
  if (param_offset->size() > CROP_OFFSET_MAX_SIZE) {
    MS_LOG(ERROR) << "parameter offset size(" << param_offset->size() << ") should <= " << CROP_OFFSET_MAX_SIZE;
    return nullptr;
  }
  CropParameter *parameter = new (std::nothrow) CropParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new CropParameter fail!";
    return nullptr;
  }
  parameter->axis_ = param->axis();
  for (int i = 0; i < param_offset->size(); ++i) {
    parameter->offset_[i] = param_offset->Get(i);
  }
  return parameter;
}

OneHotParameter *PopulateOneHotParameter(const lite::Primitive *primitive) {
  OneHotParameter *parameter = new (std::nothrow) OneHotParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new OneHotParameter fail!";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_OneHot();
  if (param == nullptr) {
    delete (parameter);
    MS_LOG(ERROR) << "get OneHot param nullptr.";
    return nullptr;
  }
  parameter->axis_ = param->axis();
  return parameter;
}

FlattenParameter *PopulateFlattenParameter(const lite::Primitive *primitive) {
  FlattenParameter *parameter = new (std::nothrow) FlattenParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new FlattenParameter fail!";
    return nullptr;
  }
  return parameter;
}

StridedSliceParameter *PopulateStridedSliceParam(const lite::Primitive *primitive) {
  StridedSliceParameter *parameter = new (std::nothrow) StridedSliceParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new StridedSliceParameter failed.";
    return nullptr;
  }
  parameter->op_parameter_.type_ = primitive->Type();
  auto n_dims = ((lite::StridedSlice *)primitive)->NDims();
  parameter->num_axes_ = n_dims;
  auto begin = ((lite::StridedSlice *)primitive)->UpdatedBegins();
  (void)memcpy(parameter->begins_, (begin.data()), begin.size() * sizeof(int));
  auto end = ((lite::StridedSlice *)primitive)->UpdatedEnds();
  (void)memcpy(parameter->ends_, (end.data()), end.size() * sizeof(int));
  auto stride = ((lite::StridedSlice *)primitive)->UpdatedStrides();
  (void)memcpy(parameter->strides_, (stride.data()), stride.size() * sizeof(int));
  auto in_shape = ((lite::StridedSlice *)primitive)->UpdatedInShape();
  (void)memcpy(parameter->in_shape_, (in_shape.data()), in_shape.size() * sizeof(int));
  return parameter;
}

OpParameter *PopulateAddNParam(const lite::Primitive *primitive) {
  auto parameter = new (std::nothrow) OpParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new OpParameter fail!";
    return nullptr;
  }
  parameter->type_ = primitive->Type();
  return parameter;
}

OpParameter *PopulateParameter(const lite::Primitive *primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_type = primitive->Type();
  switch (op_type) {
    case schema::PrimitiveType_SoftMax:
      return reinterpret_cast<OpParameter *>(PopulateSoftmaxParameter(primitive));
    case schema::PrimitiveType_Activation:
      return reinterpret_cast<OpParameter *>(PopulateActivationParameter(primitive));
    case schema::PrimitiveType_Conv2D:
      return reinterpret_cast<OpParameter *>(PopulateConvParameter(primitive));
    case schema::PrimitiveType_Reduce:
      return reinterpret_cast<OpParameter *>(PopulateReduceParameter(primitive));
    case schema::PrimitiveType_Pooling:
      return reinterpret_cast<OpParameter *>(PopulatePoolingParam(primitive));
    case schema::PrimitiveType_DepthwiseConv2D:
      return reinterpret_cast<OpParameter *>(PopulateConvDwParameter(primitive));
    case schema::PrimitiveType_DeDepthwiseConv2D:
      return reinterpret_cast<OpParameter *>(PopulateDeconvDwParameter(primitive));
    case schema::PrimitiveType_DeConv2D:
      return reinterpret_cast<OpParameter *>(PopulateDeconvParameter(primitive));
    case schema::PrimitiveType_FusedBatchNorm:
      return reinterpret_cast<OpParameter *>(PopulateFusedBatchNorm(primitive));
    case schema::PrimitiveType_FullConnection:
      return reinterpret_cast<OpParameter *>(PopulateFullconnectionParameter(primitive));
    case schema::PrimitiveType_Power:
      return reinterpret_cast<OpParameter *>(PopulatePowerParameter(primitive));
    case schema::PrimitiveType_LocalResponseNormalization:
      return reinterpret_cast<OpParameter *>(PopulateLocalResponseNormParameter(primitive));
    case schema::PrimitiveType_Range:
      return reinterpret_cast<OpParameter *>(PopulateRangeParameter(primitive));
    case schema::PrimitiveType_Transpose:
      return reinterpret_cast<OpParameter *>(PopulateTransposeParameter(primitive));
    case schema::PrimitiveType_Mul:
    case schema::PrimitiveType_Add:
    case schema::PrimitiveType_Sub:
    case schema::PrimitiveType_Div:
    case schema::PrimitiveType_FloorDiv:
    case schema::PrimitiveType_FloorMod:
    case schema::PrimitiveType_SquaredDifference:
      return reinterpret_cast<OpParameter *>(PopulateArithmetic(primitive));
    case schema::PrimitiveType_BiasAdd:
      return reinterpret_cast<OpParameter *>(new ArithmeticParameter());
    case schema::PrimitiveType_Eltwise:
      return reinterpret_cast<OpParameter *>(PopulateEltwiseParam(primitive));
    case schema::PrimitiveType_ExpandDims:
      return reinterpret_cast<OpParameter *>(PopulateExpandDimsParam(primitive));
    case schema::PrimitiveType_Abs:
    case schema::PrimitiveType_Cos:
    case schema::PrimitiveType_Sin:
    case schema::PrimitiveType_Exp:
    case schema::PrimitiveType_Log:
    case schema::PrimitiveType_Square:
    case schema::PrimitiveType_Sqrt:
    case schema::PrimitiveType_Rsqrt:
    case schema::PrimitiveType_LogicalNot:
    case schema::PrimitiveType_Floor:
      return reinterpret_cast<OpParameter *>(PopulateArithmeticSelf(primitive));
    case schema::PrimitiveType_ArgMax:
      return reinterpret_cast<OpParameter *>(PopulateArgMaxParam(primitive));
    case schema::PrimitiveType_ArgMin:
      return reinterpret_cast<OpParameter *>(PopulateArgMinParam(primitive));
    case schema::PrimitiveType_Cast:
      return reinterpret_cast<OpParameter *>(PopulateCastParam(primitive));
    case schema::PrimitiveType_Ceil:
      return reinterpret_cast<OpParameter *>(PopulateCeilParameter(primitive));
    case schema::PrimitiveType_Scale:
      return reinterpret_cast<OpParameter *>(PopulateScaleParameter(primitive));
    case schema::PrimitiveType_Reshape:
      return reinterpret_cast<OpParameter *>(PopulateReshapeParam(primitive));
    case schema::PrimitiveType_Concat:
      return reinterpret_cast<OpParameter *>(PopulateConcatParameter(primitive));
    case schema::PrimitiveType_Tile:
      return reinterpret_cast<OpParameter *>(PopulateTileParameter(primitive));
    case schema::PrimitiveType_TopK:
      return reinterpret_cast<OpParameter *>(PopulateTopKParameter(primitive));
    case schema::PrimitiveType_Fill:
      return reinterpret_cast<OpParameter *>(PopulateFillParam(primitive));
    case schema::PrimitiveType_Gather:
      return reinterpret_cast<OpParameter *>(PopulateGatherParameter(primitive));
    case schema::PrimitiveType_GatherNd:
      return reinterpret_cast<OpParameter *>(PopulateGatherNdParameter(primitive));
    case schema::PrimitiveType_Slice:
      return reinterpret_cast<OpParameter *>(PopulateSliceParam(primitive));
    case schema::PrimitiveType_BroadcastTo:
      return reinterpret_cast<OpParameter *>(PopulateBroadcastToParam(primitive));
    case schema::PrimitiveType_Reverse:
      return reinterpret_cast<OpParameter *>(PopulateReverseParameter(primitive));
    case schema::PrimitiveType_Stack:
      return reinterpret_cast<OpParameter *>(PopulateStackParam(primitive));
    case schema::PrimitiveType_Unstack:
      return reinterpret_cast<OpParameter *>(PopulateUnstackParam(primitive));
    case schema::PrimitiveType_ReverseSequence:
      return reinterpret_cast<OpParameter *>(PopulateReverseSequenceParam(primitive));
    case schema::PrimitiveType_Unique:
      return reinterpret_cast<OpParameter *>(PopulateUniqueParam(primitive));
    case schema::PrimitiveType_DepthToSpace:
      return reinterpret_cast<OpParameter *>(PopulateDepthToSpaceParam(primitive));
    case schema::PrimitiveType_Nchw2Nhwc:
      return reinterpret_cast<OpParameter *>(PopulateNchw2NhwcParameter(primitive));
    case schema::PrimitiveType_Nhwc2Nchw:
      return reinterpret_cast<OpParameter *>(PopulateNhwc2NchwParameter(primitive));
    case schema::PrimitiveType_Pad:
      return reinterpret_cast<OpParameter *>(PopulatePadParameter(primitive));
    case schema::PrimitiveType_Resize:
      return reinterpret_cast<OpParameter *>(PopulateResizeParameter(primitive));
    case schema::PrimitiveType_BatchToSpace:
      return reinterpret_cast<OpParameter *>(PopulateBatchToSpaceParameter(primitive));
    case schema::PrimitiveType_Crop:
      return reinterpret_cast<OpParameter *>(PopulateCropParameter(primitive));
    case schema::PrimitiveType_Unsqueeze:
      return reinterpret_cast<OpParameter *>(PopulateUnsqueezeParameter(primitive));
    case schema::PrimitiveType_Flatten:
      return reinterpret_cast<OpParameter *>(PopulateFlattenParameter(primitive));
    case schema::PrimitiveType_MatMul:
      return reinterpret_cast<OpParameter *>(PopulateMatMulParameter(primitive));
    case schema::PrimitiveType_OneHot:
      return reinterpret_cast<OpParameter *>(PopulateOneHotParameter(primitive));
    case schema::PrimitiveType_AddN:
      return reinterpret_cast<OpParameter *>(PopulateAddNParam(primitive));
    default:
      break;
  }
  return nullptr;
}
}  // namespace mindspore::kernel
