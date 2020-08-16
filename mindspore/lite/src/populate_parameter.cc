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
#include "schema/ops_generated.h"
#include "src/runtime/kernel/arm/nnacl/op_base.h"
#include "src/runtime/kernel/arm/nnacl/fp32/arg_min_max.h"
#include "src/runtime/kernel/arm/nnacl/fp32/cast.h"
#include "src/runtime/kernel/arm/nnacl/concat_parameter.h"
#include "src/runtime/kernel/arm/nnacl/caffeprelu_parameter.h"
#include "src/runtime/kernel/arm/nnacl/fp32/slice.h"
#include "src/runtime/kernel/arm/nnacl/fp32/broadcast_to.h"
#include "src/runtime/kernel/arm/nnacl/reshape_parameter.h"
#include "src/runtime/kernel/arm/nnacl/shape.h"
#include "src/runtime/kernel/arm/nnacl/fp32/stack.h"
#include "src/runtime/kernel/arm/nnacl/unstack.h"
#include "src/runtime/kernel/arm/nnacl/depth_to_space.h"
#include "src/runtime/kernel/arm/nnacl/conv_parameter.h"
#include "src/runtime/kernel/arm/nnacl/fp32/pooling.h"
#include "src/runtime/kernel/arm/nnacl/matmul_parameter.h"
#include "src/runtime/kernel/arm/nnacl/fp32/roi_pooling.h"
#include "src/runtime/kernel/arm/nnacl/softmax_parameter.h"
#include "src/runtime/kernel/arm/nnacl/fp32/tile.h"
#include "src/runtime/kernel/arm/nnacl/fp32/topk.h"
#include "src/runtime/kernel/arm/nnacl/reduce_parameter.h"
#include "src/runtime/kernel/arm/nnacl/fp32/activation.h"
#include "src/runtime/kernel/arm/nnacl/fp32/arithmetic.h"
#include "src/runtime/kernel/arm/nnacl/fp32/batchnorm.h"
#include "src/runtime/kernel/arm/nnacl/power.h"
#include "src/runtime/kernel/arm/nnacl/fp32/range.h"
#include "src/runtime/kernel/arm/nnacl/fp32/local_response_norm.h"
#include "src/runtime/kernel/arm/nnacl/fp32/expandDims.h"
#include "src/runtime/kernel/arm/nnacl/arithmetic_self_parameter.h"
#include "src/runtime/kernel/arm/nnacl/pad_parameter.h"
#include "src/runtime/kernel/arm/nnacl/fp32/fill.h"
#include "src/runtime/kernel/arm/nnacl/transpose.h"
#include "src/runtime/kernel/arm/nnacl/split_parameter.h"
#include "src/runtime/kernel/arm/nnacl/squeeze.h"
#include "src/runtime/kernel/arm/nnacl/fp32/gather.h"
#include "src/runtime/kernel/arm/nnacl/fp32/reverse.h"
#include "src/runtime/kernel/arm/nnacl/reverse_sequence.h"
#include "src/runtime/kernel/arm/nnacl/fp32/unique.h"
#include "src/runtime/kernel/arm/nnacl/scale.h"
#include "src/runtime/kernel/arm/nnacl/fp32/gatherNd.h"
#include "src/runtime/kernel/arm/nnacl/resize_parameter.h"
#include "src/runtime/kernel/arm/nnacl/scatter_nd.h"
#include "src/runtime/kernel/arm/nnacl/batch_to_space.h"
#include "src/runtime/kernel/arm/nnacl/fp32/crop.h"
#include "src/runtime/kernel/arm/fp32/flatten.h"
#include "src/runtime/kernel/arm/nnacl/fp32/unsqueeze.h"
#include "src/runtime/kernel/arm/nnacl/fp32/one_hot.h"
#include "src/runtime/kernel/arm/nnacl/strided_slice.h"
#include "src/runtime/kernel/arm/base/prior_box.h"
#include "src/runtime/kernel/arm/nnacl/fp32/space_to_depth.h"
#include "src/runtime/kernel/arm/nnacl/fp32/space_to_batch.h"
#include "src/runtime/kernel/arm/nnacl/int8/quant_dtype_cast.h"
#include "src/runtime/kernel/arm/nnacl/fp32/lstm.h"
#include "src/runtime/kernel/arm/nnacl/fp32/embedding_lookup.h"
#include "src/runtime/kernel/arm/nnacl/fp32/elu.h"
#include "src/runtime/kernel/arm/nnacl/prelu_parameter.h"

namespace mindspore::kernel {

OpParameter *PopulateROIPoolingParameter(const lite::Primitive *primitive) {
  auto pooling_primitive = primitive->Value()->value_as_ROIPooling();
  ROIPoolingParameter *param = new (std::nothrow) ROIPoolingParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "new PoolingParameter failed.";
    return nullptr;
  }
  param->op_parameter_.type_ = primitive->Type();
  param->pooledH_ = pooling_primitive->pooledH();
  param->pooledW_ = pooling_primitive->pooledW();
  param->scale_ = pooling_primitive->scale();
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateBatchNorm(const lite::Primitive *primitive) {
  BatchNormParameter *batch_norm_param = new (std::nothrow) BatchNormParameter();
  if (batch_norm_param == nullptr) {
    MS_LOG(ERROR) << "new BatchNormParameter failed.";
    return nullptr;
  }
  batch_norm_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_BatchNorm();
  batch_norm_param->epsilon_ = param->epsilon();
  return reinterpret_cast<OpParameter *>(batch_norm_param);
}

OpParameter *PopulateFillParameter(const lite::Primitive *primitive) {
  auto param = primitive->Value()->value_as_Fill();
  FillParameter *fill_param = new (std::nothrow) FillParameter();
  if (fill_param == nullptr) {
    MS_LOG(ERROR) << "new FillParameter failed.";
    return nullptr;
  }
  fill_param->op_parameter_.type_ = primitive->Type();
  auto flatDims = param->dims();
  fill_param->num_dims_ = flatDims->size();
  int i = 0;
  for (auto iter = flatDims->begin(); iter != flatDims->end(); iter++) {
    fill_param->dims_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(fill_param);
}

OpParameter *PopulateExpandDimsParameter(const lite::Primitive *primitive) {
  auto param = primitive->Value()->value_as_ExpandDims();
  ExpandDimsParameter *expand_dims_param = new (std::nothrow) ExpandDimsParameter();
  if (expand_dims_param == nullptr) {
    MS_LOG(ERROR) << "new ExpandDimsParameter failed.";
    return nullptr;
  }
  expand_dims_param->op_parameter_.type_ = primitive->Type();
  expand_dims_param->dim_ = param->dim();
  return reinterpret_cast<OpParameter *>(expand_dims_param);
}

OpParameter *PopulateCaffePReLUParameter(const lite::Primitive *primitive) {
  auto param = primitive->Value()->value_as_CaffePReLU();
  CaffePreluParameter *caffePrelu_param = new (std::nothrow) CaffePreluParameter();
  if (caffePrelu_param == nullptr) {
    MS_LOG(ERROR) << "new caffePReluParameter failed.";
    return nullptr;
  }
  caffePrelu_param->op_parameter_.type_ = primitive->Type();
  caffePrelu_param->channelShared = param->channelShared();
  return reinterpret_cast<OpParameter *>(caffePrelu_param);
}

OpParameter *PopulatePreluParameter(const lite::Primitive *primitive) {
  auto param = primitive->Value()->value_as_Prelu();
  PreluParameter *Prelu_param = new (std::nothrow) PreluParameter();
  if (Prelu_param == nullptr) {
    MS_LOG(ERROR) << "new caffePReluParameter failed.";
    return nullptr;
  }
  Prelu_param->op_parameter_.type_ = primitive->Type();
  auto temp = param->slope();
  for (int i = 0; i < temp->size(); i++) {
    Prelu_param->slope_[i] = temp->Get(i);
  }
  return reinterpret_cast<OpParameter *>(Prelu_param);
}

OpParameter *PopulatePoolingParameter(const lite::Primitive *primitive) {
  auto pooling_primitive = primitive->Value()->value_as_Pooling();
  // todo use malloc instead
  PoolingParameter *pooling_param = new (std::nothrow) PoolingParameter();
  if (pooling_param == nullptr) {
    MS_LOG(ERROR) << "new PoolingParameter failed.";
    return nullptr;
  }
  pooling_param->op_parameter_.type_ = primitive->Type();
  pooling_param->global_ = pooling_primitive->global();
  pooling_param->window_w_ = pooling_primitive->windowW();
  pooling_param->window_h_ = pooling_primitive->windowH();
  auto pooling_lite_primitive = (lite::Pooling *)primitive;
  MS_ASSERT(nullptr != pooling_lite_primitive);
  pooling_param->pad_u_ = pooling_lite_primitive->PadUp();
  pooling_param->pad_d_ = pooling_lite_primitive->PadDown();
  pooling_param->pad_l_ = pooling_lite_primitive->PadLeft();
  pooling_param->pad_r_ = pooling_lite_primitive->PadRight();
  pooling_param->stride_w_ = pooling_primitive->strideW();
  pooling_param->stride_h_ = pooling_primitive->strideH();

  auto is_global = pooling_primitive->global();
  pooling_param->global_ = is_global;
  auto pool_mode = pooling_primitive->poolingMode();
  switch (pool_mode) {
    case schema::PoolMode_MAX_POOLING:
      pooling_param->max_pooling_ = true;
      pooling_param->avg_pooling_ = false;
      break;
    case schema::PoolMode_MEAN_POOLING:
      pooling_param->max_pooling_ = false;
      pooling_param->avg_pooling_ = true;
      break;
    default:
      pooling_param->max_pooling_ = false;
      pooling_param->avg_pooling_ = false;
      break;
  }

  auto round_mode = pooling_primitive->roundMode();
  switch (round_mode) {
    case schema::RoundMode_FLOOR:
      pooling_param->round_floor_ = true;
      pooling_param->round_ceil_ = false;
      break;
    case schema::RoundMode_CEIL:
      pooling_param->round_floor_ = false;
      pooling_param->round_ceil_ = true;
      break;
    default:
      pooling_param->round_floor_ = false;
      pooling_param->round_ceil_ = false;
      break;
  }
  return reinterpret_cast<OpParameter *>(pooling_param);
}

OpParameter *PopulateFullconnectionParameter(const lite::Primitive *primitive) {
  auto param = primitive->Value()->value_as_FullConnection();
  MatMulParameter *matmul_param = new (std::nothrow) MatMulParameter();
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "new FullconnectionParameter failed.";
    return nullptr;
  }
  matmul_param->op_parameter_.type_ = primitive->Type();
  matmul_param->b_transpose_ = true;
  matmul_param->a_transpose_ = false;
  matmul_param->has_bias_ = param->hasBias();
  if (param->activationType() == schema::ActivationType_RELU) {
    matmul_param->act_type_ = ActType_Relu;
  } else if (param->activationType() == schema::ActivationType_RELU6) {
    matmul_param->act_type_ = ActType_Relu6;
  } else {
    matmul_param->act_type_ = ActType_No;
  }

  return reinterpret_cast<OpParameter *>(matmul_param);
}

OpParameter *PopulateMatMulParameter(const lite::Primitive *primitive) {
  auto param = primitive->Value()->value_as_MatMul();
  MatMulParameter *matmul_param = new (std::nothrow) MatMulParameter();
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "new FullconnectionParameter failed.";
    return nullptr;
  }
  matmul_param->op_parameter_.type_ = primitive->Type();
  matmul_param->b_transpose_ = param->transposeB();
  matmul_param->a_transpose_ = param->transposeA();
  matmul_param->has_bias_ = false;
  matmul_param->act_type_ = ActType_No;
  return reinterpret_cast<OpParameter *>(matmul_param);
}

OpParameter *PopulateConvParameter(const lite::Primitive *primitive) {
  ConvParameter *conv_param = new (std::nothrow) ConvParameter();
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }
  conv_param->op_parameter_.type_ = primitive->Type();
  auto conv_primitive = primitive->Value()->value_as_Conv2D();
  conv_param->kernel_h_ = conv_primitive->kernelH();
  conv_param->kernel_w_ = conv_primitive->kernelW();
  // todo format
  conv_param->group_ = conv_primitive->group();
  conv_param->stride_h_ = conv_primitive->strideH();
  conv_param->stride_w_ = conv_primitive->strideW();

  auto conv2d_lite_primitive = (lite::Conv2D *)primitive;
  MS_ASSERT(nullptr != conv2d_lite_primitive);
  conv_param->pad_u_ = conv2d_lite_primitive->PadUp();
  conv_param->pad_d_ = conv2d_lite_primitive->PadDown();
  conv_param->pad_l_ = conv2d_lite_primitive->PadLeft();
  conv_param->pad_r_ = conv2d_lite_primitive->PadRight();
  conv_param->pad_h_ = conv2d_lite_primitive->PadUp();
  conv_param->pad_w_ = conv2d_lite_primitive->PadLeft();
  conv_param->dilation_h_ = conv_primitive->dilateH();
  conv_param->dilation_w_ = conv_primitive->dilateW();
  conv_param->input_channel_ = conv_primitive->channelIn();
  conv_param->output_channel_ = conv_primitive->channelOut();
  conv_param->group_ = conv_primitive->group();
  auto act_type = conv_primitive->activationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      conv_param->is_relu_ = true;
      conv_param->is_relu6_ = false;
      break;
    case schema::ActivationType_RELU6:
      conv_param->is_relu_ = false;
      conv_param->is_relu6_ = true;
      break;
    default:
      conv_param->is_relu_ = false;
      conv_param->is_relu6_ = false;
      break;
  }
  return reinterpret_cast<OpParameter *>(conv_param);
}

OpParameter *PopulateConvDwParameter(const lite::Primitive *primitive) {
  ConvParameter *conv_param = new (std::nothrow) ConvParameter();
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }
  conv_param->op_parameter_.type_ = primitive->Type();
  auto conv_primitive = primitive->Value()->value_as_DepthwiseConv2D();
  conv_param->kernel_h_ = conv_primitive->kernelH();
  conv_param->kernel_w_ = conv_primitive->kernelW();
  // todo format, group
  conv_param->stride_h_ = conv_primitive->strideH();
  conv_param->stride_w_ = conv_primitive->strideW();

  auto pad_mode = conv_primitive->padMode();
  auto convdw_lite_primitive = (lite::DepthwiseConv2D *)primitive;
  MS_ASSERT(nullptr != convdw_lite_primitive);
  conv_param->pad_u_ = convdw_lite_primitive->PadUp();
  conv_param->pad_d_ = convdw_lite_primitive->PadDown();
  conv_param->pad_l_ = convdw_lite_primitive->PadLeft();
  conv_param->pad_r_ = convdw_lite_primitive->PadRight();
  conv_param->pad_h_ = convdw_lite_primitive->PadUp();
  conv_param->pad_w_ = convdw_lite_primitive->PadLeft();
  conv_param->dilation_h_ = conv_primitive->dilateH();
  conv_param->dilation_w_ = conv_primitive->dilateW();
  auto act_type = conv_primitive->activationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      conv_param->is_relu_ = true;
      conv_param->is_relu6_ = false;
      break;
    case schema::ActivationType_RELU6:
      conv_param->is_relu_ = false;
      conv_param->is_relu6_ = true;
      break;
    default:
      conv_param->is_relu_ = false;
      conv_param->is_relu6_ = false;
      break;
  }
  return reinterpret_cast<OpParameter *>(conv_param);
}

OpParameter *PopulateDeconvDwParameter(const lite::Primitive *primitive) {
  ConvParameter *conv_param = new ConvParameter();
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }
  conv_param->op_parameter_.type_ = primitive->Type();
  auto conv_primitive = primitive->Value()->value_as_DeDepthwiseConv2D();
  conv_param->kernel_h_ = conv_primitive->kernelH();
  conv_param->kernel_w_ = conv_primitive->kernelW();
  // todo format, group
  conv_param->stride_h_ = conv_primitive->strideH();
  conv_param->stride_w_ = conv_primitive->strideW();

  auto deconvdw_lite_primitive = (lite::DeconvDepthwiseConv2D *)primitive;
  MS_ASSERT(nullptr != deconvdw_lite_primitive);
  conv_param->pad_u_ = deconvdw_lite_primitive->PadUp();
  conv_param->pad_d_ = deconvdw_lite_primitive->PadDown();
  conv_param->pad_l_ = deconvdw_lite_primitive->PadLeft();
  conv_param->pad_r_ = deconvdw_lite_primitive->PadRight();
  conv_param->pad_h_ = deconvdw_lite_primitive->PadUp();
  conv_param->pad_w_ = deconvdw_lite_primitive->PadLeft();
  conv_param->dilation_h_ = conv_primitive->dilateH();
  conv_param->dilation_w_ = conv_primitive->dilateW();
  auto act_type = conv_primitive->activationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      conv_param->is_relu_ = true;
      conv_param->is_relu6_ = false;
      break;
    case schema::ActivationType_RELU6:
      conv_param->is_relu_ = false;
      conv_param->is_relu6_ = true;
      break;
    default:
      conv_param->is_relu_ = false;
      conv_param->is_relu6_ = false;
      break;
  }
  return reinterpret_cast<OpParameter *>(conv_param);
}

OpParameter *PopulateDeconvParameter(const lite::Primitive *primitive) {
  ConvParameter *conv_param = new ConvParameter();
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }
  conv_param->op_parameter_.type_ = primitive->Type();
  auto conv_primitive = primitive->Value()->value_as_DeConv2D();
  conv_param->kernel_h_ = conv_primitive->kernelH();
  conv_param->kernel_w_ = conv_primitive->kernelW();
  conv_param->stride_h_ = conv_primitive->strideH();
  conv_param->stride_w_ = conv_primitive->strideW();

  auto deconv_lite_primitive = (lite::DeConv2D *)primitive;
  MS_ASSERT(nullptr != deconvdw_lite_primitive);
  conv_param->pad_u_ = deconv_lite_primitive->PadUp();
  conv_param->pad_d_ = deconv_lite_primitive->PadDown();
  conv_param->pad_l_ = deconv_lite_primitive->PadLeft();
  conv_param->pad_r_ = deconv_lite_primitive->PadRight();
  conv_param->dilation_h_ = conv_primitive->dilateH();
  conv_param->dilation_w_ = conv_primitive->dilateW();
  auto act_type = conv_primitive->activationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      conv_param->is_relu_ = true;
      conv_param->is_relu6_ = false;
      break;
    case schema::ActivationType_RELU6:
      conv_param->is_relu_ = false;
      conv_param->is_relu6_ = true;
      break;
    default:
      conv_param->is_relu_ = false;
      conv_param->is_relu6_ = false;
      break;
  }

  auto pad_mode = conv_primitive->padMode();
  switch (pad_mode) {
    case schema::PadMode_SAME:
      conv_param->pad_h_ = (conv_param->kernel_h_ - 1) / 2;
      conv_param->pad_w_ = (conv_param->kernel_w_ - 1) / 2;
      break;
    case schema::PadMode_VALID:
      conv_param->pad_h_ = 0;
      conv_param->pad_w_ = 0;
      break;
    case schema::PadMode_CAFFE:
      conv_param->pad_h_ = conv_param->pad_u_;
      conv_param->pad_w_ = conv_param->pad_l_;
      break;
    default:
      MS_LOG(ERROR) << "invalid pad mode!";
      return nullptr;
  }

  return reinterpret_cast<OpParameter *>(conv_param);
}

OpParameter *PopulateSoftmaxParameter(const lite::Primitive *primitive) {
  auto softmax_primitive = primitive->Value()->value_as_SoftMax();
  SoftmaxParameter *softmax_param = new (std::nothrow) SoftmaxParameter();
  if (softmax_param == nullptr) {
    MS_LOG(ERROR) << "new SoftmaxParameter failed.";
    return nullptr;
  }
  softmax_param->op_parameter_.type_ = primitive->Type();
  softmax_param->axis_ = softmax_primitive->axis();
  return reinterpret_cast<OpParameter *>(softmax_param);
}

OpParameter *PopulateReduceParameter(const lite::Primitive *primitive) {
  ReduceParameter *reduce_param = new (std::nothrow) ReduceParameter();
  if (reduce_param == nullptr) {
    MS_LOG(ERROR) << "new ReduceParameter failed.";
    return nullptr;
  }
  reduce_param->op_parameter_.type_ = primitive->Type();
  auto reduce = primitive->Value()->value_as_Reduce();
  reduce_param->keep_dims_ = reduce->keepDims();
  auto axisVector = reduce->axes();
  if (axisVector->size() > REDUCE_MAX_AXES_NUM) {
    MS_LOG(ERROR) << "Reduce axes size " << axisVector->size() << " exceed limit " << REDUCE_MAX_AXES_NUM;
    delete (reduce_param);
    return nullptr;
  }
  reduce_param->num_axes_ = static_cast<int>(axisVector->size());
  int i = 0;
  for (auto iter = axisVector->begin(); iter != axisVector->end(); iter++) {
    reduce_param->axes_[i++] = *iter;
  }
  reduce_param->mode_ = static_cast<int>(reduce->mode());
  return reinterpret_cast<OpParameter *>(reduce_param);
}

OpParameter *PopulateMeanParameter(const lite::Primitive *primitive) {
  ReduceParameter *mean_param = new (std::nothrow) ReduceParameter();
  if (mean_param == nullptr) {
    MS_LOG(ERROR) << "new ReduceParameter failed.";
    return nullptr;
  }
  mean_param->op_parameter_.type_ = primitive->Type();
  auto mean = primitive->Value()->value_as_Mean();
  mean_param->keep_dims_ = mean->keepDims();
  auto axisVector = mean->axis();
  if (axisVector->size() > REDUCE_MAX_AXES_NUM) {
    MS_LOG(ERROR) << "Reduce axes size " << axisVector->size() << " exceed limit " << REDUCE_MAX_AXES_NUM;
    delete (mean_param);
    return nullptr;
  }
  mean_param->num_axes_ = static_cast<int>(axisVector->size());
  int i = 0;
  for (auto iter = axisVector->begin(); iter != axisVector->end(); iter++) {
    mean_param->axes_[i++] = *iter;
  }
  mean_param->mode_ = static_cast<int>(schema::ReduceMode_ReduceMean);
  return reinterpret_cast<OpParameter *>(mean_param);
}

OpParameter *PopulatePadParameter(const lite::Primitive *primitive) {
  PadParameter *pad_param = new (std::nothrow) PadParameter();
  if (pad_param == nullptr) {
    MS_LOG(ERROR) << "new PadParameter failed.";
    return nullptr;
  }
  pad_param->op_parameter_.type_ = primitive->Type();
  auto pad_node = primitive->Value()->value_as_Pad();
  pad_param->pad_mode_ = pad_node->paddingMode();
  if (pad_param->pad_mode_ == schema::PaddingMode_CONSTANT) {
    pad_param->constant_value_ = pad_node->constantValue();
  } else {
    MS_LOG(ERROR) << "Invalid padding mode: " << pad_param->pad_mode_;
    delete (pad_param);
    return nullptr;
  }

  auto size = pad_node->paddings()->size();
  if (size > MAX_PAD_SIZE) {
    MS_LOG(ERROR) << "Invalid padding size: " << size;
    delete (pad_param);
    return nullptr;
  }

  for (size_t i = 0; i < size; i++) {
    pad_param->paddings_[MAX_PAD_SIZE - size + i] = (*(pad_node->paddings()))[i];
  }
  return reinterpret_cast<OpParameter *>(pad_param);
}

OpParameter *PopulateActivationParameter(const lite::Primitive *primitive) {
  ActivationParameter *act_param = new (std::nothrow) ActivationParameter();
  if (act_param == nullptr) {
    MS_LOG(ERROR) << "new ActivationParameter failed.";
    return nullptr;
  }
  auto activation = primitive->Value()->value_as_Activation();
  act_param->type_ = static_cast<int>(activation->type());
  act_param->alpha_ = activation->alpha();
  return reinterpret_cast<OpParameter *>(act_param);
}

OpParameter *PopulateFusedBatchNorm(const lite::Primitive *primitive) {
  BatchNormParameter *batch_norm_param = new (std::nothrow) BatchNormParameter();
  if (batch_norm_param == nullptr) {
    MS_LOG(ERROR) << "new FusedBatchNormParameter failed.";
    return nullptr;
  }
  batch_norm_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_FusedBatchNorm();
  batch_norm_param->epsilon_ = param->epsilon();
  return reinterpret_cast<OpParameter *>(batch_norm_param);
}

OpParameter *PopulateArithmetic(const lite::Primitive *primitive) {
  ArithmeticParameter *arithmetic_param = new (std::nothrow) ArithmeticParameter();
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticParameter failed.";
    return nullptr;
  }
  arithmetic_param->op_parameter_.type_ = primitive->Type();
  arithmetic_param->broadcasting_ = ((lite::Arithmetic *)primitive)->Broadcasting();
  arithmetic_param->ndim_ = ((lite::Arithmetic *)primitive)->NDims();
  switch (primitive->Type()) {
    case schema::PrimitiveType_Add:
      arithmetic_param->activation_type_ = primitive->Value()->value_as_Add()->activationType();
      break;
    case schema::PrimitiveType_Sub:
      arithmetic_param->activation_type_ = primitive->Value()->value_as_Sub()->activationType();
      break;
    case schema::PrimitiveType_Mul:
      arithmetic_param->activation_type_ = primitive->Value()->value_as_Mul()->activationType();
      break;
    case schema::PrimitiveType_Div:
      arithmetic_param->activation_type_ = primitive->Value()->value_as_Div()->activationType();
      break;
    default:
      arithmetic_param->activation_type_ = 0;
      break;
  }
  auto tmp_shape = ((lite::Arithmetic *)primitive)->InShape0();
  (void)memcpy(arithmetic_param->in_shape0_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  tmp_shape = ((lite::Arithmetic *)primitive)->InShape1();
  (void)memcpy(arithmetic_param->in_shape1_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  tmp_shape = ((lite::Arithmetic *)primitive)->OutputShape();
  (void)memcpy(arithmetic_param->out_shape_, static_cast<void *>(tmp_shape.data()), tmp_shape.size() * sizeof(int));
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

OpParameter *PopulateEltwiseParameter(const lite::Primitive *primitive) {
  ArithmeticParameter *arithmetic_param = new (std::nothrow) ArithmeticParameter();
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticParameter failed.";
    return nullptr;
  }
  auto eltwise = primitive->Value()->value_as_Eltwise();
  switch (eltwise->mode()) {
    case schema::EltwiseMode_PROD:
      arithmetic_param->op_parameter_.type_ = schema::PrimitiveType_Mul;
      break;
    case schema::EltwiseMode_SUM:
      arithmetic_param->op_parameter_.type_ = schema::PrimitiveType_Add;
      break;
    case schema::EltwiseMode_MAXIMUM:
      arithmetic_param->op_parameter_.type_ = schema::PrimitiveType_Maximum;
      break;
    default:
      delete arithmetic_param;
      return nullptr;
  }
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

OpParameter *PopulateArithmeticSelf(const lite::Primitive *primitive) {
  ArithmeticSelfParameter *arithmetic_self_param = new (std::nothrow) ArithmeticSelfParameter();
  if (arithmetic_self_param == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticParameter failed.";
    return nullptr;
  }
  arithmetic_self_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(arithmetic_self_param);
}

OpParameter *PopulatePowerParameter(const lite::Primitive *primitive) {
  PowerParameter *power_param = new (std::nothrow) PowerParameter();
  if (power_param == nullptr) {
    MS_LOG(ERROR) << "new PowerParameter failed.";
    return nullptr;
  }
  power_param->op_parameter_.type_ = primitive->Type();
  auto power = primitive->Value()->value_as_Power();
  power_param->power_ = power->power();
  power_param->scale_ = power->scale();
  power_param->shift_ = power->shift();
  return reinterpret_cast<OpParameter *>(power_param);
}

OpParameter *PopulateArgMaxParameter(const lite::Primitive *primitive) {
  ArgMinMaxParameter *arg_param = new (std::nothrow) ArgMinMaxParameter();
  if (arg_param == nullptr) {
    MS_LOG(ERROR) << "new ArgMinMaxParameter failed.";
    return nullptr;
  }
  arg_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_ArgMax();
  arg_param->axis_ = param->axis();
  arg_param->topk_ = param->topK();
  arg_param->axis_type_ = param->axisType();
  arg_param->out_value_ = param->outMaxValue();
  arg_param->keep_dims_ = param->keepDims();
  return reinterpret_cast<OpParameter *>(arg_param);
}

OpParameter *PopulateArgMinParameter(const lite::Primitive *primitive) {
  ArgMinMaxParameter *arg_param = new (std::nothrow) ArgMinMaxParameter();
  if (arg_param == nullptr) {
    MS_LOG(ERROR) << "new ArgMinMaxParameter failed.";
    return nullptr;
  }
  arg_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_ArgMin();
  arg_param->axis_ = param->axis();
  arg_param->topk_ = param->topK();
  arg_param->axis_type_ = param->axisType();
  arg_param->out_value_ = param->outMaxValue();
  arg_param->keep_dims_ = param->keepDims();
  return reinterpret_cast<OpParameter *>(arg_param);
}

OpParameter *PopulateCastParameter(const lite::Primitive *primitive) {
  CastParameter *cast_param = new (std::nothrow) CastParameter();
  if (cast_param == nullptr) {
    MS_LOG(ERROR) << "new CastParameter failed.";
    return nullptr;
  }
  cast_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_Cast();
  cast_param->src_type_ = param->srcT();
  cast_param->dst_type_ = param->dstT();
  return reinterpret_cast<OpParameter *>(cast_param);
}

OpParameter *PopulateLocalResponseNormParameter(const lite::Primitive *primitive) {
  auto local_response_norm_attr = primitive->Value()->value_as_LocalResponseNormalization();
  LocalResponseNormParameter *lrn_param = new (std::nothrow) LocalResponseNormParameter();
  if (lrn_param == nullptr) {
    MS_LOG(ERROR) << "new LocalResponseNormParameter failed.";
    return nullptr;
  }
  lrn_param->op_parameter_.type_ = primitive->Type();
  lrn_param->depth_radius_ = local_response_norm_attr->depth_radius();
  lrn_param->bias_ = local_response_norm_attr->bias();
  lrn_param->alpha_ = local_response_norm_attr->alpha();
  lrn_param->beta_ = local_response_norm_attr->beta();
  return reinterpret_cast<OpParameter *>(lrn_param);
}

OpParameter *PopulateRangeParameter(const lite::Primitive *primitive) {
  auto range_attr = primitive->Value()->value_as_Range();
  RangeParameter *range_param = new (std::nothrow) RangeParameter();
  if (range_param == nullptr) {
    MS_LOG(ERROR) << "new RangeParameter failed.";
    return nullptr;
  }
  range_param->op_parameter_.type_ = primitive->Type();
  range_param->start_ = range_attr->start();
  range_param->limit_ = range_attr->limit();
  range_param->delta_ = range_attr->delta();
  range_param->dType_ = range_attr->dType();
  return reinterpret_cast<OpParameter *>(range_param);
}

OpParameter *PopulateConcatParameter(const lite::Primitive *primitive) {
  ConcatParameter *concat_param = new (std::nothrow) ConcatParameter();
  if (concat_param == nullptr) {
    MS_LOG(ERROR) << "new ConcatParameter failed.";
    return nullptr;
  }
  concat_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_Concat();
  concat_param->axis_ = param->axis();
  return reinterpret_cast<OpParameter *>(concat_param);
}

OpParameter *PopulateTileParameter(const lite::Primitive *primitive) {
  TileParameter *tile_param = new (std::nothrow) TileParameter();
  if (tile_param == nullptr) {
    MS_LOG(ERROR) << "new TileParameter failed.";
    return nullptr;
  }
  tile_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_Tile();
  auto multiples = param->multiples();
  tile_param->in_dim_ = multiples->size();
  for (size_t i = 0; i < tile_param->in_dim_; ++i) {
    tile_param->multiples_[i] = multiples->Get(i);
  }
  return reinterpret_cast<OpParameter *>(tile_param);
}

OpParameter *PopulateTopKParameter(const lite::Primitive *primitive) {
  TopkParameter *topk_param = new (std::nothrow) TopkParameter();
  if (topk_param == nullptr) {
    MS_LOG(ERROR) << "new TopkParameter failed.";
    return nullptr;
  }
  topk_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_TopK();
  topk_param->k_ = param->k();
  topk_param->sorted_ = param->sorted();
  return reinterpret_cast<OpParameter *>(topk_param);
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

OpParameter *PopulateTransposeParameter(const lite::Primitive *primitive) {
  TransposeParameter *transpose_param = new (std::nothrow) TransposeParameter();
  if (transpose_param == nullptr) {
    MS_LOG(ERROR) << "new TransposeParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_Transpose();
  transpose_param->op_parameter_.type_ = primitive->Type();
  auto perm_vector_ = param->perm();
  int i = 0;
  for (auto iter = perm_vector_->begin(); iter != perm_vector_->end(); iter++) {
    transpose_param->perm_[i++] = *iter;
  }
  transpose_param->num_axes_ = i;
  transpose_param->conjugate_ = param->conjugate();
  return reinterpret_cast<OpParameter *>(transpose_param);
}

OpParameter *PopulateSplitParameter(const lite::Primitive *primitive) {
  SplitParameter *split_param = new (std::nothrow) SplitParameter();
  if (split_param == nullptr) {
    MS_LOG(ERROR) << "new SplitParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_Split();
  split_param->op_parameter_.type_ = primitive->Type();
  split_param->num_split_ = param->numberSplit();
  auto split_sizes_vector_ = param->sizeSplits();
  int i = 0;
  for (auto iter = split_sizes_vector_->begin(); iter != split_sizes_vector_->end(); iter++) {
    split_param->split_sizes_[i++] = *iter;
  }
  split_param->split_dim_ = param->splitDim();
  split_param->num_split_ = param->numberSplit();
  return reinterpret_cast<OpParameter *>(split_param);
}

OpParameter *PopulateSqueezeParameter(const lite::Primitive *primitive) {
  SqueezeParameter *squeeze_param = new (std::nothrow) SqueezeParameter();
  if (squeeze_param == nullptr) {
    MS_LOG(ERROR) << "new SqueezeParameter failed.";
    return nullptr;
  }
  squeeze_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(squeeze_param);
}

OpParameter *PopulateScaleParameter(const lite::Primitive *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "input primitive is nullptr";
    return nullptr;
  }
  ScaleParameter *scale_param = new (std::nothrow) ScaleParameter();
  if (scale_param == nullptr) {
    MS_LOG(ERROR) << "new ScaleParameter failed.";
    return nullptr;
  }
  scale_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_Scale();
  if (param == nullptr) {
    MS_LOG(ERROR) << "value_as_Scale return nullptr";
    return nullptr;
  }
  scale_param->axis_ = param->axis();
  return reinterpret_cast<OpParameter *>(scale_param);
}

OpParameter *PopulateGatherParameter(const lite::Primitive *primitive) {
  auto gather_attr = primitive->Value()->value_as_Gather();
  GatherParameter *gather_param = new (std::nothrow) GatherParameter();
  if (gather_param == nullptr) {
    MS_LOG(ERROR) << "new GatherParameter failed.";
    return nullptr;
  }
  gather_param->op_parameter_.type_ = primitive->Type();
  gather_param->axis_ = gather_attr->axis();
  gather_param->batchDims_ = gather_attr->batchDims();
  return reinterpret_cast<OpParameter *>(gather_param);
}

OpParameter *PopulateGatherNdParameter(const lite::Primitive *primitive) {
  GatherNdParameter *gather_nd_param = new (std::nothrow) GatherNdParameter();
  if (gather_nd_param == nullptr) {
    MS_LOG(ERROR) << "new GatherNDParameter failed.";
    return nullptr;
  }
  gather_nd_param->op_parameter_.type_ = primitive->Type();
  auto gatherNd_attr = primitive->Value()->value_as_GatherNd();
  gather_nd_param->batchDims_ = gatherNd_attr->batchDims();
  return reinterpret_cast<OpParameter *>(gather_nd_param);
}

OpParameter *PopulateScatterNDParameter(const lite::Primitive *primitive) {
  ScatterNDParameter *scatter_nd_param = new (std::nothrow) ScatterNDParameter();
  if (scatter_nd_param == nullptr) {
    MS_LOG(ERROR) << "new ScatterNDParameter failed.";
    return nullptr;
  }
  scatter_nd_param->op_parameter_.type_ = primitive->Type();
  MS_ASSERT(paramter != nullptr);
  return reinterpret_cast<OpParameter *>(scatter_nd_param);
}

OpParameter *PopulateSliceParameter(const lite::Primitive *primitive) {
  SliceParameter *slice_param = new (std::nothrow) SliceParameter();
  if (slice_param == nullptr) {
    MS_LOG(ERROR) << "new SliceParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_Slice();
  slice_param->op_parameter_.type_ = primitive->Type();
  auto param_begin = param->begin();
  auto param_size = param->size();
  if (param_begin->size() != param_size->size()) {
    delete slice_param;
    return nullptr;
  }
  slice_param->param_length_ = static_cast<int32_t>(param_begin->size());
  for (int32_t i = 0; i < slice_param->param_length_; ++i) {
    slice_param->begin_[i] = param_begin->Get(i);
    slice_param->size_[i] = param_size->Get(i);
  }
  return reinterpret_cast<OpParameter *>(slice_param);
}

OpParameter *PopulateBroadcastToParameter(const lite::Primitive *primitive) {
  BroadcastToParameter *broadcast_param = new (std::nothrow) BroadcastToParameter();
  if (broadcast_param == nullptr) {
    MS_LOG(ERROR) << "new BroadcastToParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_BroadcastTo();
  broadcast_param->op_parameter_.type_ = primitive->Type();
  auto dst_shape = param->dst_shape();
  broadcast_param->shape_size_ = dst_shape->size();
  for (size_t i = 0; i < broadcast_param->shape_size_; ++i) {
    broadcast_param->shape_[i] = dst_shape->Get(i);
  }
  return reinterpret_cast<OpParameter *>(broadcast_param);
}

OpParameter *PopulateReshapeParameter(const lite::Primitive *primitive) {
  ReshapeParameter *reshape_param = new (std::nothrow) ReshapeParameter();
  if (reshape_param == nullptr) {
    MS_LOG(ERROR) << "new ReshapeParameter failed.";
    return nullptr;
  }
  reshape_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(reshape_param);
}

OpParameter *PopulateShapeParameter(const lite::Primitive *primitive) {
  ShapeParameter *shape_param = new (std::nothrow) ShapeParameter();
  if (shape_param == nullptr) {
    MS_LOG(ERROR) << "new ShapeParameter failed.";
    return nullptr;
  }
  shape_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(shape_param);
}

OpParameter *PopulateReverseParameter(const lite::Primitive *primitive) {
  auto reverse_attr = primitive->Value()->value_as_Reverse();
  ReverseParameter *reverse_param = new (std::nothrow) ReverseParameter();
  if (reverse_param == nullptr) {
    MS_LOG(ERROR) << "new ReverseParameter failed.";
    return nullptr;
  }
  reverse_param->op_parameter_.type_ = primitive->Type();
  auto flatAxis = reverse_attr->axis();
  reverse_param->num_axis_ = flatAxis->size();
  int i = 0;
  for (auto iter = flatAxis->begin(); iter != flatAxis->end(); iter++) {
    reverse_param->axis_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(reverse_param);
}

OpParameter *PopulateUnsqueezeParameter(const lite::Primitive *primitive) {
  auto unsqueeze_attr = primitive->Value()->value_as_Unsqueeze();
  UnsqueezeParameter *unsqueeze_param = new (std::nothrow) UnsqueezeParameter();
  if (unsqueeze_param == nullptr) {
    MS_LOG(ERROR) << "new ReverseParameter failed.";
    return nullptr;
  }
  unsqueeze_param->op_parameter_.type_ = primitive->Type();
  auto flatAxis = unsqueeze_attr->axis();
  unsqueeze_param->num_dim_ = flatAxis->size();
  int i = 0;
  for (auto iter = flatAxis->begin(); iter != flatAxis->end(); iter++) {
    unsqueeze_param->dims_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(unsqueeze_param);
}

OpParameter *PopulateStackParameter(const lite::Primitive *primitive) {
  StackParameter *stack_param = new (std::nothrow) StackParameter();
  if (stack_param == nullptr) {
    MS_LOG(ERROR) << "new StackParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_Stack();
  stack_param->op_parameter_.type_ = primitive->Type();
  stack_param->axis_ = param->axis();
  return reinterpret_cast<OpParameter *>(stack_param);
}

OpParameter *PopulateUnstackParameter(const lite::Primitive *primitive) {
  UnstackParameter *unstack_param = new (std::nothrow) UnstackParameter();
  if (unstack_param == nullptr) {
    MS_LOG(ERROR) << "new UnstackParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_Unstack();
  unstack_param->op_parameter_.type_ = primitive->Type();
  unstack_param->num_ = param->num();
  unstack_param->axis_ = param->axis();
  return reinterpret_cast<OpParameter *>(unstack_param);
}

OpParameter *PopulateReverseSequenceParameter(const lite::Primitive *primitive) {
  ReverseSequenceParameter *reverse_sequence_param = new (std::nothrow) ReverseSequenceParameter();
  if (reverse_sequence_param == nullptr) {
    MS_LOG(ERROR) << "new ReverseSequenceParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_ReverseSequence();
  reverse_sequence_param->op_parameter_.type_ = primitive->Type();
  reverse_sequence_param->seq_axis_ = param->seqAxis();
  reverse_sequence_param->batch_axis_ = param->batchAxis();
  return reinterpret_cast<OpParameter *>(reverse_sequence_param);
}

OpParameter *PopulateUniqueParameter(const lite::Primitive *primitive) {
  UniqueParameter *unique_param = new (std::nothrow) UniqueParameter();
  if (unique_param == nullptr) {
    MS_LOG(ERROR) << "new PopulateUniqueParam failed.";
    return nullptr;
  }
  unique_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(unique_param);
}

OpParameter *PopulateDepthToSpaceParameter(const lite::Primitive *primitive) {
  DepthToSpaceParameter *depth_space_param = new (std::nothrow) DepthToSpaceParameter();
  if (depth_space_param == nullptr) {
    MS_LOG(ERROR) << "new DepthToSpaceParameter failed.";
    return nullptr;
  }
  auto param = primitive->Value()->value_as_DepthToSpace();
  depth_space_param->op_parameter_.type_ = primitive->Type();
  depth_space_param->block_size_ = param->blockSize();
  return reinterpret_cast<OpParameter *>(depth_space_param);
}

OpParameter *PopulateSpaceToDepthParameter(const lite::Primitive *primitive) {
  SpaceToDepthParameter *space_depth_param = new (std::nothrow) SpaceToDepthParameter();
  if (space_depth_param == nullptr) {
    MS_LOG(ERROR) << "new SpaceToDepthspace_depth_param failed.";
    return nullptr;
  }
  space_depth_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_DepthToSpace();
  space_depth_param->op_parameter_.type_ = primitive->Type();
  space_depth_param->block_size_ = param->blockSize();
  if (param->format() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "Currently only NHWC format is supported.";
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(space_depth_param);
}

OpParameter *PopulateSpaceToBatchParameter(const lite::Primitive *primitive) {
  SpaceToBatchParameter *space_batch_param = new (std::nothrow) SpaceToBatchParameter();
  if (space_batch_param == nullptr) {
    MS_LOG(ERROR) << "new SpaceToBatchParameter failed.";
    return nullptr;
  }
  space_batch_param->op_parameter_.type_ = primitive->Type();
  space_batch_param->op_parameter_.type_ = primitive->Type();
  auto block_sizes = ((lite::SpaceToBatch *)primitive)->BlockSizes();
  (void)memcpy(space_batch_param->block_sizes_, (block_sizes.data()), block_sizes.size() * sizeof(int));
  auto paddings = ((lite::SpaceToBatch *)primitive)->Paddings();
  (void)memcpy(space_batch_param->paddings_, (paddings.data()), paddings.size() * sizeof(int));
  auto in_shape = ((lite::SpaceToBatch *)primitive)->InShape();
  (void)memcpy(space_batch_param->in_shape_, (in_shape.data()), in_shape.size() * sizeof(int));
  auto padded_in_shape = ((lite::SpaceToBatch *)primitive)->PaddedInShape();
  (void)memcpy(space_batch_param->padded_in_shape_, (padded_in_shape.data()), padded_in_shape.size() * sizeof(int));
  return reinterpret_cast<OpParameter *>(space_batch_param);
}

OpParameter *PopulateResizeParameter(const lite::Primitive *primitive) {
  ResizeParameter *resize_param = new (std::nothrow) ResizeParameter();
  if (resize_param == nullptr) {
    MS_LOG(ERROR) << "new ResizeParameter failed.";
    return nullptr;
  }
  resize_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_Resize();
  resize_param->method_ = static_cast<int>(param->method());
  resize_param->new_height_ = param->newHeight();
  resize_param->new_width_ = param->newWidth();
  resize_param->align_corners_ = param->alignCorners();
  resize_param->preserve_aspect_ratio_ = param->preserveAspectRatio();
  return reinterpret_cast<OpParameter *>(resize_param);
}

OpParameter *PopulateBatchToSpaceParameter(const lite::Primitive *primitive) {
  BatchToSpaceParameter *batch_space_param = new (std::nothrow) BatchToSpaceParameter();
  if (batch_space_param == nullptr) {
    MS_LOG(ERROR) << "New BatchToSpaceParameter fail!";
    return nullptr;
  }
  batch_space_param->op_parameter_.type_ = primitive->Type();
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
    batch_space_param->block_shape_[i] = block_shape->Get(i);
  }

  for (int i = 0; i < BATCH_TO_SPACE_CROPS_SIZE; ++i) {
    batch_space_param->crops_[i] = crops->Get(i);
  }
  return reinterpret_cast<OpParameter *>(batch_space_param);
}

OpParameter *PopulateCropParameter(const lite::Primitive *primitive) {
  auto param = primitive->Value()->value_as_Crop();
  auto param_offset = param->offsets();
  if (param_offset->size() > CROP_OFFSET_MAX_SIZE) {
    MS_LOG(ERROR) << "crop_param offset size(" << param_offset->size() << ") should <= " << CROP_OFFSET_MAX_SIZE;
    return nullptr;
  }
  CropParameter *crop_param = new (std::nothrow) CropParameter();
  if (crop_param == nullptr) {
    MS_LOG(ERROR) << "new CropParameter fail!";
    return nullptr;
  }
  crop_param->op_parameter_.type_ = primitive->Type();
  crop_param->axis_ = param->axis();
  crop_param->offset_size_ = param_offset->size();
  for (int i = 0; i < param_offset->size(); ++i) {
    crop_param->offset_[i] = param_offset->Get(i);
  }
  return reinterpret_cast<OpParameter *>(crop_param);
}

OpParameter *PopulateOneHotParameter(const lite::Primitive *primitive) {
  OneHotParameter *one_hot_param = new (std::nothrow) OneHotParameter();
  if (one_hot_param == nullptr) {
    MS_LOG(ERROR) << "new OneHotParameter fail!";
    return nullptr;
  }
  one_hot_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_OneHot();
  if (param == nullptr) {
    delete (one_hot_param);
    MS_LOG(ERROR) << "get OneHot param nullptr.";
    return nullptr;
  }
  one_hot_param->axis_ = param->axis();
  return reinterpret_cast<OpParameter *>(one_hot_param);
}

OpParameter *PopulateFlattenParameter(const lite::Primitive *primitive) {
  FlattenParameter *flatten_param = new (std::nothrow) FlattenParameter();
  if (flatten_param == nullptr) {
    MS_LOG(ERROR) << "new FlattenParameter fail!";
    return nullptr;
  }
  flatten_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(flatten_param);
}

OpParameter *PopulateQuantDTypeCastParameter(const lite::Primitive *primitive) {
  QuantDTypeCastParameter *parameter = new (std::nothrow) QuantDTypeCastParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new QuantDTypeCastParameter fail!";
    return nullptr;
  }
  parameter->op_parameter_.type_ = primitive->Type();
  auto quant_dtype_cast_param = primitive->Value()->value_as_QuantDTypeCast();
  parameter->srcT = quant_dtype_cast_param->srcT();
  parameter->dstT = quant_dtype_cast_param->dstT();
  return reinterpret_cast<OpParameter *>(parameter);
}

OpParameter *PopulateStridedSliceParameter(const lite::Primitive *primitive) {
  StridedSliceParameter *strided_slice_param = new (std::nothrow) StridedSliceParameter();
  if (strided_slice_param == nullptr) {
    MS_LOG(ERROR) << "new StridedSliceParameter failed.";
    return nullptr;
  }
  strided_slice_param->op_parameter_.type_ = primitive->Type();
  auto n_dims = ((lite::StridedSlice *)primitive)->NDims();
  strided_slice_param->num_axes_ = n_dims;
  auto begin = ((lite::StridedSlice *)primitive)->GetBegins();
  (void)memcpy(strided_slice_param->begins_, (begin.data()), begin.size() * sizeof(int));
  auto end = ((lite::StridedSlice *)primitive)->GetEnds();
  (void)memcpy(strided_slice_param->ends_, (end.data()), end.size() * sizeof(int));
  auto stride = ((lite::StridedSlice *)primitive)->GetStrides();
  (void)memcpy(strided_slice_param->strides_, (stride.data()), stride.size() * sizeof(int));
  auto in_shape = ((lite::StridedSlice *)primitive)->GetInShape();
  (void)memcpy(strided_slice_param->in_shape_, (in_shape.data()), in_shape.size() * sizeof(int));
  return reinterpret_cast<OpParameter *>(strided_slice_param);
}

OpParameter *PopulateAddNParameter(const lite::Primitive *primitive) {
  auto addn_param = new (std::nothrow) OpParameter();
  if (addn_param == nullptr) {
    MS_LOG(ERROR) << "new OpParameter fail!";
    return nullptr;
  }
  addn_param->type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(addn_param);
}

OpParameter *PopulatePriorBoxParameter(const lite::Primitive *primitive) {
  PriorBoxParameter *prior_box_param = new (std::nothrow) PriorBoxParameter();
  if (prior_box_param == nullptr) {
    MS_LOG(ERROR) << "new PriorBoxParameter failed.";
    return nullptr;
  }
  prior_box_param->op_parameter_.type_ = primitive->Type();
  auto prior_box_attr = primitive->Value()->value_as_PriorBox();

  if (prior_box_attr->min_sizes()->size() > PRIOR_BOX_MAX_NUM) {
    MS_LOG(ERROR) << "PriorBox min_sizes size exceeds max num " << PRIOR_BOX_MAX_NUM << ", got "
                  << prior_box_attr->min_sizes();
    delete (prior_box_param);
    return nullptr;
  }
  prior_box_param->min_sizes_size = prior_box_attr->min_sizes()->size();
  if (prior_box_attr->max_sizes()->size() > PRIOR_BOX_MAX_NUM) {
    MS_LOG(ERROR) << "PriorBox max_sizes size exceeds max num " << PRIOR_BOX_MAX_NUM << ", got "
                  << prior_box_attr->max_sizes();
    delete (prior_box_param);
    return nullptr;
  }
  prior_box_param->max_sizes_size = prior_box_attr->max_sizes()->size();
  (void)memcpy(prior_box_param->max_sizes, prior_box_attr->max_sizes()->data(),
               prior_box_attr->max_sizes()->size() * sizeof(int32_t));
  (void)memcpy(prior_box_param->min_sizes, prior_box_attr->min_sizes()->data(),
               prior_box_attr->min_sizes()->size() * sizeof(int32_t));

  if (prior_box_attr->aspect_ratios()->size() > PRIOR_BOX_MAX_NUM) {
    MS_LOG(ERROR) << "PriorBox aspect_ratios size exceeds max num " << PRIOR_BOX_MAX_NUM << ", got "
                  << prior_box_attr->aspect_ratios();
    delete (prior_box_param);
    return nullptr;
  }
  prior_box_param->aspect_ratios_size = prior_box_attr->aspect_ratios()->size();
  (void)memcpy(prior_box_param->aspect_ratios, prior_box_attr->aspect_ratios()->data(),
               prior_box_attr->aspect_ratios()->size() * sizeof(float));
  if (prior_box_attr->variances()->size() != PRIOR_BOX_VAR_NUM) {
    MS_LOG(ERROR) << "PriorBox variances size should be " << PRIOR_BOX_VAR_NUM << ", got "
                  << prior_box_attr->variances()->size();
    delete (prior_box_param);
    return nullptr;
  }
  (void)memcpy(prior_box_param->variances, prior_box_attr->variances()->data(), PRIOR_BOX_VAR_NUM * sizeof(float));
  prior_box_param->flip = prior_box_attr->flip();
  prior_box_param->clip = prior_box_attr->clip();
  prior_box_param->offset = prior_box_attr->offset();
  prior_box_param->image_size_h = prior_box_attr->image_size_h();
  prior_box_param->image_size_w = prior_box_attr->image_size_w();
  prior_box_param->step_h = prior_box_attr->step_h();
  prior_box_param->step_w = prior_box_attr->step_w();
  return reinterpret_cast<OpParameter *>(prior_box_param);
}

OpParameter *PopulateLstmParameter(const lite::Primitive *primitive) {
  LstmParameter *lstm_param = new (std::nothrow) LstmParameter();
  if (lstm_param == nullptr) {
    MS_LOG(ERROR) << "new LstmParameter fail!";
    return nullptr;
  }
  lstm_param->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_Lstm();
  if (param == nullptr) {
    delete (lstm_param);
    MS_LOG(ERROR) << "get Lstm param nullptr.";
    return nullptr;
  }
  lstm_param->bidirectional_ = param->bidirection();
  return reinterpret_cast<OpParameter *>(lstm_param);
}

OpParameter *PopulateEmbeddingLookupParameter(const lite::Primitive *primitive) {
  EmbeddingLookupParameter *embedding_lookup_parameter = new (std::nothrow) EmbeddingLookupParameter();
  if (embedding_lookup_parameter == nullptr) {
    MS_LOG(ERROR) << "new EmbeddingLookupParameter failed";
    return nullptr;
  }
  embedding_lookup_parameter->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_EmbeddingLookup();
  embedding_lookup_parameter->max_norm_ = param->maxNorm();
  if (embedding_lookup_parameter->max_norm_ < 0) {
    MS_LOG(ERROR) << "Embedding lookup max norm should be positive number, got "
                  << embedding_lookup_parameter->max_norm_;
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(embedding_lookup_parameter);
}

OpParameter *PopulateBiasAddParameter(const lite::Primitive *primitive) {
  ArithmeticParameter *arithmetic_param = new (std::nothrow) ArithmeticParameter();
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "new Bias Add Parameter failed";
    return nullptr;
  }
  arithmetic_param->op_parameter_.type_ = primitive->Type();

  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

OpParameter *PopulateEluParameter(const lite::Primitive *primitive) {
  EluParameter *elu_parameter = new (std::nothrow) EluParameter();
  if (elu_parameter == nullptr) {
    MS_LOG(ERROR) << "new EluParameter failed";
    return nullptr;
  }
  elu_parameter->op_parameter_.type_ = primitive->Type();
  auto param = primitive->Value()->value_as_Elu();
  elu_parameter->alpha_ = param->alpha();
  return reinterpret_cast<OpParameter *>(elu_parameter);
}

PopulateParameterRegistry::PopulateParameterRegistry() {
  populate_parameter_funcs_[schema::PrimitiveType_SoftMax] = PopulateSoftmaxParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Activation] = PopulateActivationParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Conv2D] = PopulateConvParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Reduce] = PopulateReduceParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Mean] = PopulateMeanParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Pooling] = PopulatePoolingParameter;
  populate_parameter_funcs_[schema::PrimitiveType_ROIPooling] = PopulateROIPoolingParameter;
  populate_parameter_funcs_[schema::PrimitiveType_DepthwiseConv2D] = PopulateConvDwParameter;
  populate_parameter_funcs_[schema::PrimitiveType_DeDepthwiseConv2D] = PopulateDeconvDwParameter;
  populate_parameter_funcs_[schema::PrimitiveType_DeConv2D] = PopulateDeconvParameter;
  populate_parameter_funcs_[schema::PrimitiveType_FusedBatchNorm] = PopulateFusedBatchNorm;
  populate_parameter_funcs_[schema::PrimitiveType_BatchNorm] = PopulateBatchNorm;
  populate_parameter_funcs_[schema::PrimitiveType_FullConnection] = PopulateFullconnectionParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Power] = PopulatePowerParameter;
  populate_parameter_funcs_[schema::PrimitiveType_LocalResponseNormalization] = PopulateLocalResponseNormParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Range] = PopulateRangeParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Transpose] = PopulateTransposeParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Mul] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_Add] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_Sub] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_Div] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_LogicalAnd] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_LogicalOr] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_Equal] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_Less] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_Greater] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_NotEqual] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_LessEqual] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_GreaterEqual] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_Maximum] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_Minimum] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_FloorDiv] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_FloorMod] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_SquaredDifference] = PopulateArithmetic;
  populate_parameter_funcs_[schema::PrimitiveType_BiasAdd] = PopulateBiasAddParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Eltwise] = PopulateEltwiseParameter;
  populate_parameter_funcs_[schema::PrimitiveType_ExpandDims] = PopulateExpandDimsParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Abs] = PopulateArithmeticSelf;
  populate_parameter_funcs_[schema::PrimitiveType_Cos] = PopulateArithmeticSelf;
  populate_parameter_funcs_[schema::PrimitiveType_Sin] = PopulateArithmeticSelf;
  populate_parameter_funcs_[schema::PrimitiveType_Exp] = PopulateArithmeticSelf;
  populate_parameter_funcs_[schema::PrimitiveType_Log] = PopulateArithmeticSelf;
  populate_parameter_funcs_[schema::PrimitiveType_Square] = PopulateArithmeticSelf;
  populate_parameter_funcs_[schema::PrimitiveType_Sqrt] = PopulateArithmeticSelf;
  populate_parameter_funcs_[schema::PrimitiveType_Rsqrt] = PopulateArithmeticSelf;
  populate_parameter_funcs_[schema::PrimitiveType_LogicalNot] = PopulateArithmeticSelf;
  populate_parameter_funcs_[schema::PrimitiveType_Floor] = PopulateArithmeticSelf;
  populate_parameter_funcs_[schema::PrimitiveType_Ceil] = PopulateArithmeticSelf;
  populate_parameter_funcs_[schema::PrimitiveType_Round] = PopulateArithmeticSelf;
  populate_parameter_funcs_[schema::PrimitiveType_ArgMax] = PopulateArgMaxParameter;
  populate_parameter_funcs_[schema::PrimitiveType_ArgMin] = PopulateArgMinParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Cast] = PopulateCastParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Scale] = PopulateScaleParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Reshape] = PopulateReshapeParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Shape] = PopulateShapeParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Concat] = PopulateConcatParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Tile] = PopulateTileParameter;
  populate_parameter_funcs_[schema::PrimitiveType_TopK] = PopulateTopKParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Fill] = PopulateFillParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Gather] = PopulateGatherParameter;
  populate_parameter_funcs_[schema::PrimitiveType_GatherNd] = PopulateGatherNdParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Slice] = PopulateSliceParameter;
  populate_parameter_funcs_[schema::PrimitiveType_BroadcastTo] = PopulateBroadcastToParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Reverse] = PopulateReverseParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Stack] = PopulateStackParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Unstack] = PopulateUnstackParameter;
  populate_parameter_funcs_[schema::PrimitiveType_ReverseSequence] = PopulateReverseSequenceParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Unique] = PopulateUniqueParameter;
  populate_parameter_funcs_[schema::PrimitiveType_DepthToSpace] = PopulateDepthToSpaceParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Nchw2Nhwc] = PopulateNchw2NhwcParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Nhwc2Nchw] = PopulateNhwc2NchwParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Pad] = PopulatePadParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Resize] = PopulateResizeParameter;
  populate_parameter_funcs_[schema::PrimitiveType_BatchToSpace] = PopulateBatchToSpaceParameter;
  populate_parameter_funcs_[schema::PrimitiveType_SpaceToDepth] = PopulateSpaceToDepthParameter;
  populate_parameter_funcs_[schema::PrimitiveType_SpaceToBatch] = PopulateSpaceToBatchParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Crop] = PopulateCropParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Unsqueeze] = PopulateUnsqueezeParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Flatten] = PopulateFlattenParameter;
  populate_parameter_funcs_[schema::PrimitiveType_MatMul] = PopulateMatMulParameter;
  populate_parameter_funcs_[schema::PrimitiveType_OneHot] = PopulateOneHotParameter;
  populate_parameter_funcs_[schema::PrimitiveType_AddN] = PopulateAddNParameter;
  populate_parameter_funcs_[schema::PrimitiveType_StridedSlice] = PopulateStridedSliceParameter;
  populate_parameter_funcs_[schema::PrimitiveType_ScatterND] = PopulateScatterNDParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Squeeze] = PopulateSqueezeParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Split] = PopulateSplitParameter;
  populate_parameter_funcs_[schema::PrimitiveType_CaffePReLU] = PopulateCaffePReLUParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Prelu] = PopulatePreluParameter;
  populate_parameter_funcs_[schema::PrimitiveType_PriorBox] = PopulatePriorBoxParameter;
  populate_parameter_funcs_[schema::PrimitiveType_QuantDTypeCast] = PopulateQuantDTypeCastParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Lstm] = PopulateLstmParameter;
  populate_parameter_funcs_[schema::PrimitiveType_EmbeddingLookup] = PopulateEmbeddingLookupParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Elu] = PopulateEluParameter;
}

PopulateParameterRegistry *PopulateParameterRegistry::GetInstance() {
  static PopulateParameterRegistry populate_parameter_instance;
  return &populate_parameter_instance;
}

PopulateParameterFunc PopulateParameterRegistry::GetParameterFunc(const schema::PrimitiveType &type) {
  return populate_parameter_funcs_[type];
}

OpParameter *PopulateParameter(const lite::Primitive *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }

  auto op_type = primitive->Type();
  auto func = PopulateParameterRegistry::GetInstance()->GetParameterFunc(op_type);
  if (func == nullptr) {
    MS_LOG(ERROR) << "Get nullptr for Op Parameter Func.";
    return nullptr;
  }

  auto *parameter = func(primitive);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "Get nullptr for Op Parameter.";
    return nullptr;
  }
  return parameter;
}
}  // namespace mindspore::kernel
