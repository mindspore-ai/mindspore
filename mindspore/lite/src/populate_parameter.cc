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
#include "src/ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "schema/ops_generated.h"
#include "src/ops/constant_of_shape.h"
#include "src/ops/space_to_batch.h"
#include "src/ops/conv2d.h"
#include "src/ops/roi_pooling.h"
#include "src/ops/topk.h"
#include "src/ops/broadcast_to.h"
#include "src/ops/unsqueeze.h"
#include "src/ops/unstack.h"
#include "src/ops/depth_to_space.h"
#include "src/ops/batch_to_space.h"
#include "src/ops/prior_box.h"
#include "src/ops/lstm.h"
#include "src/ops/softmax.h"
#include "src/ops/activation.h"
#include "src/ops/deconv2d.h"
#include "src/ops/reduce.h"
#include "src/ops/pooling.h"
#include "src/ops/fused_batchnorm.h"
#include "src/ops/batch_norm.h"
#include "src/ops/power.h"
#include "src/ops/range.h"
#include "src/ops/add.h"
#include "src/ops/sub.h"
#include "src/ops/div.h"
#include "src/ops/bias_add.h"
#include "src/ops/expand_dims.h"
#include "src/ops/full_connection.h"
#include "src/ops/shape.h"
#include "src/ops/elu.h"
#include "src/ops/embedding_lookup.h"
#include "src/ops/quant_dtype_cast.h"
#include "src/ops/matmul.h"
#include "src/ops/resize.h"
#include "src/ops/tile.h"
#include "src/ops/one_hot.h"
#include "src/ops/space_to_depth.h"
#include "src/ops/split.h"
#include "src/ops/argmax.h"
#include "src/ops/argmin.h"
#include "src/ops/cast.h"
#include "src/ops/reshape.h"
#include "src/ops/scale.h"
#include "src/ops/concat.h"
#include "src/ops/nchw2nhwc.h"
#include "src/ops/slice.h"
#include "src/ops/squeeze.h"
#include "src/ops/flatten.h"
#include "src/ops/mean.h"
#include "src/ops/nhwc2nchw.h"
#include "src/ops/stack.h"
#include "src/ops/crop.h"
#include "src/ops/addn.h"
#include "src/ops/gather.h"
#include "src/ops/gather_nd.h"
#include "src/ops/local_response_normalization.h"
#include "src/ops/pad.h"
#include "src/ops/prelu.h"
#include "src/ops/caffe_p_relu.h"
#include "src/ops/reverse_sequence.h"
#include "src/ops/dedepthwise_conv2d.h"
#include "src/ops/depthwise_conv2d.h"
#include "src/ops/mul.h"
#include "src/ops/eltwise.h"
#include "src/ops/fill.h"
#include "src/ops/transpose.h"
#include "src/ops/log.h"
#include "src/ops/abs.h"
#include "src/ops/sin.h"
#include "src/ops/cos.h"
#include "src/ops/sqrt.h"
#include "src/ops/square.h"
#include "src/ops/exp.h"
#include "src/ops/rsqrt.h"
#include "src/ops/maximum.h"
#include "src/ops/minimum.h"
#include "src/ops/strided_slice.h"
#include "src/ops/reverse.h"
#include "src/ops/logical_and.h"
#include "src/ops/logical_or.h"
#include "src/ops/logical_not.h"
#include "src/ops/floor_div.h"
#include "src/ops/floor_mod.h"
#include "src/ops/equal.h"
#include "src/ops/not_equal.h"
#include "src/ops/less.h"
#include "src/ops/less_equal.h"
#include "src/ops/greater_equal.h"
#include "src/ops/greater.h"
#include "src/ops/floor.h"
#include "src/ops/squared_difference.h"
#include "src/ops/ceil.h"
#include "src/ops/round.h"
#include "nnacl/op_base.h"
#include "nnacl/fp32/arg_min_max.h"
#include "nnacl/fp32/cast.h"
#include "nnacl/concat_parameter.h"
#include "nnacl/fp32/slice.h"
#include "nnacl/fp32/broadcast_to.h"
#include "nnacl/reshape_parameter.h"
#include "nnacl/prelu_parameter.h"
#include "nnacl/shape.h"
#include "nnacl/fp32/constant_of_shape.h"
#include "nnacl/fp32/stack.h"
#include "nnacl/unstack.h"
#include "nnacl/depth_to_space.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/fp32/pooling.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/fp32/roi_pooling.h"
#include "nnacl/softmax_parameter.h"
#include "nnacl/fp32/tile.h"
#include "nnacl/fp32/topk.h"
#include "nnacl/reduce_parameter.h"
#include "nnacl/fp32/activation.h"
#include "nnacl/fp32/arithmetic.h"
#include "nnacl/fp32/batchnorm.h"
#include "nnacl/power.h"
#include "nnacl/fp32/range.h"
#include "nnacl/fp32/local_response_norm.h"
#include "nnacl/fp32/expandDims.h"
#include "nnacl/arithmetic_self_parameter.h"
#include "nnacl/pad_parameter.h"
#include "nnacl/fp32/fill.h"
#include "nnacl/transpose.h"
#include "nnacl/split_parameter.h"
#include "nnacl/squeeze.h"
#include "nnacl/gather_parameter.h"
#include "nnacl/fp32/reverse.h"
#include "nnacl/reverse_sequence.h"
#include "nnacl/fp32/unique.h"
#include "nnacl/scale.h"
#include "nnacl/fp32/gatherNd.h"
#include "nnacl/resize_parameter.h"
#include "nnacl/scatter_nd.h"
#include "nnacl/batch_to_space.h"
#include "nnacl/fp32/crop.h"
#include "fp32/flatten.h"
#include "nnacl/fp32/unsqueeze.h"
#include "nnacl/fp32/one_hot.h"
#include "nnacl/strided_slice.h"
#include "base/prior_box.h"
#include "nnacl/fp32/space_to_depth.h"
#include "nnacl/fp32/space_to_batch.h"
#include "nnacl/int8/quant_dtype_cast.h"
#include "nnacl/fp32/lstm.h"
#include "nnacl/fp32/embedding_lookup.h"
#include "nnacl/fp32/elu.h"
#include "nnacl/leaky_relu_parameter.h"

namespace mindspore::kernel {

OpParameter *PopulateROIPoolingParameter(const mindspore::lite::PrimitiveC *primitive) {
  const auto param =
    reinterpret_cast<mindspore::lite::ROIPooling *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  ROIPoolingParameter *roi_pooling_param = reinterpret_cast<ROIPoolingParameter *>(malloc(sizeof(ROIPoolingParameter)));
  if (roi_pooling_param == nullptr) {
    MS_LOG(ERROR) << "malloc ROIPoolingParameter failed.";
    return nullptr;
  }
  memset(roi_pooling_param, 0, sizeof(ROIPoolingParameter));
  roi_pooling_param->op_parameter_.type_ = primitive->Type();
  roi_pooling_param->pooledH_ = param->GetPooledW();
  roi_pooling_param->pooledW_ = param->GetPooledW();
  roi_pooling_param->scale_ = param->GetScale();
  return reinterpret_cast<OpParameter *>(roi_pooling_param);
}

OpParameter *PopulateBatchNorm(const mindspore::lite::PrimitiveC *primitive) {
  const auto param =
    reinterpret_cast<mindspore::lite::BatchNorm *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  BatchNormParameter *batch_norm_param = reinterpret_cast<BatchNormParameter *>(malloc(sizeof(BatchNormParameter)));
  if (batch_norm_param == nullptr) {
    MS_LOG(ERROR) << "malloc BatchNormParameter failed.";
    return nullptr;
  }
  memset(batch_norm_param, 0, sizeof(BatchNormParameter));
  batch_norm_param->op_parameter_.type_ = primitive->Type();
  batch_norm_param->epsilon_ = param->GetEpsilon();
  batch_norm_param->fused_ = false;
  return reinterpret_cast<OpParameter *>(batch_norm_param);
}

OpParameter *PopulateFillParameter(const mindspore::lite::PrimitiveC *primitive) {
  const auto param = reinterpret_cast<mindspore::lite::Fill *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  FillParameter *fill_param = reinterpret_cast<FillParameter *>(malloc(sizeof(FillParameter)));
  if (fill_param == nullptr) {
    MS_LOG(ERROR) << "malloc FillParameter failed.";
    return nullptr;
  }
  memset(fill_param, 0, sizeof(FillParameter));
  fill_param->op_parameter_.type_ = primitive->Type();
  auto flatDims = param->GetDims();
  fill_param->num_dims_ = flatDims.size();
  int i = 0;
  for (auto iter = flatDims.begin(); iter != flatDims.end(); iter++) {
    fill_param->dims_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(fill_param);
}

OpParameter *PopulateExpandDimsParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto param = reinterpret_cast<mindspore::lite::ExpandDims *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  ExpandDimsParameter *expand_dims_param = reinterpret_cast<ExpandDimsParameter *>(malloc(sizeof(ExpandDimsParameter)));
  if (expand_dims_param == nullptr) {
    MS_LOG(ERROR) << "malloc ExpandDimsParameter failed.";
    return nullptr;
  }
  memset(expand_dims_param, 0, sizeof(ExpandDimsParameter));
  expand_dims_param->op_parameter_.type_ = primitive->Type();
  expand_dims_param->dim_ = param->GetDim();
  return reinterpret_cast<OpParameter *>(expand_dims_param);
}

OpParameter *PopulatePReLUParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto param = dynamic_cast<const mindspore::lite::CaffePReLU *>(primitive);
  PReluParameter *prelu_param = reinterpret_cast<PReluParameter *>(malloc(sizeof(PReluParameter)));
  if (prelu_param == nullptr) {
    MS_LOG(ERROR) << "malloc PReluParameter failed.";
    return nullptr;
  }
  memset(prelu_param, 0, sizeof(PReluParameter));
  prelu_param->op_parameter_.type_ = primitive->Type();
  prelu_param->channelShared = param->GetChannelShared();
  return reinterpret_cast<OpParameter *>(prelu_param);
}

OpParameter *PopulateLeakyReluParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto param = dynamic_cast<const mindspore::lite::Prelu *>(primitive);
  LeakyReluParameter *leaky_relu_param = reinterpret_cast<LeakyReluParameter *>(malloc(sizeof(LeakyReluParameter)));
  if (leaky_relu_param == nullptr) {
    MS_LOG(ERROR) << "malloc LeakyReluParameter failed.";
    return nullptr;
  }
  memset(leaky_relu_param, 0, sizeof(LeakyReluParameter));
  leaky_relu_param->op_parameter_.type_ = primitive->Type();
  auto temp = param->GetSlope();
  leaky_relu_param->slope_ = reinterpret_cast<float *>(malloc(temp.size() * sizeof(float)));
  if (leaky_relu_param->slope_ == nullptr) {
    MS_LOG(ERROR) << "malloc relu slope fail!";
    free(leaky_relu_param);
    return nullptr;
  }
  for (size_t i = 0; i < temp.size(); i++) {
    leaky_relu_param->slope_[i] = temp[i];
  }
  leaky_relu_param->slope_num_ = temp.size();
  return reinterpret_cast<OpParameter *>(leaky_relu_param);
}

OpParameter *PopulatePoolingParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto pooling_primitive =
    reinterpret_cast<mindspore::lite::Pooling *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  PoolingParameter *pooling_param = reinterpret_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  if (pooling_param == nullptr) {
    MS_LOG(ERROR) << "malloc PoolingParameter failed.";
    return nullptr;
  }
  memset(pooling_param, 0, sizeof(PoolingParameter));
  pooling_param->op_parameter_.type_ = primitive->Type();
  pooling_param->global_ = pooling_primitive->GetGlobal();
  pooling_param->window_w_ = pooling_primitive->GetWindowW();
  pooling_param->window_h_ = pooling_primitive->GetWindowH();
  auto pooling_lite_primitive = (lite::Pooling *)primitive;
  pooling_param->pad_u_ = pooling_lite_primitive->PadUp();
  pooling_param->pad_d_ = pooling_lite_primitive->PadDown();
  pooling_param->pad_l_ = pooling_lite_primitive->PadLeft();
  pooling_param->pad_r_ = pooling_lite_primitive->PadRight();
  pooling_param->stride_w_ = pooling_primitive->GetStrideW();
  pooling_param->stride_h_ = pooling_primitive->GetStrideH();

  auto is_global = pooling_primitive->GetGlobal();
  pooling_param->global_ = is_global;
  auto pool_mode = pooling_primitive->GetPoolingMode();
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

  auto round_mode = pooling_primitive->GetRoundMode();
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

OpParameter *PopulateFullconnectionParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto param =
    reinterpret_cast<mindspore::lite::FullConnection *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  MatMulParameter *matmul_param = reinterpret_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "malloc MatMulParameter failed.";
    return nullptr;
  }
  memset(matmul_param, 0, sizeof(MatMulParameter));
  matmul_param->op_parameter_.type_ = primitive->Type();
  matmul_param->b_transpose_ = true;
  matmul_param->a_transpose_ = false;
  matmul_param->has_bias_ = param->GetHasBias();
  if (param->GetActivationType() == schema::ActivationType_RELU) {
    matmul_param->act_type_ = ActType_Relu;
  } else if (param->GetActivationType() == schema::ActivationType_RELU6) {
    matmul_param->act_type_ = ActType_Relu6;
  } else {
    matmul_param->act_type_ = ActType_No;
  }

  return reinterpret_cast<OpParameter *>(matmul_param);
}

OpParameter *PopulateMatMulParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto param = reinterpret_cast<mindspore::lite::MatMul *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  MatMulParameter *matmul_param = reinterpret_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "malloc MatMulParameter failed.";
    return nullptr;
  }
  memset(matmul_param, 0, sizeof(MatMulParameter));
  matmul_param->op_parameter_.type_ = primitive->Type();
  matmul_param->b_transpose_ = param->GetTransposeB();
  matmul_param->a_transpose_ = param->GetTransposeA();
  matmul_param->has_bias_ = false;
  matmul_param->act_type_ = ActType_No;
  return reinterpret_cast<OpParameter *>(matmul_param);
}

OpParameter *PopulateConvParameter(const mindspore::lite::PrimitiveC *primitive) {
  ConvParameter *conv_param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "malloc ConvParameter failed.";
    return nullptr;
  }
  memset(conv_param, 0, sizeof(ConvParameter));
  conv_param->op_parameter_.type_ = primitive->Type();
  auto conv_primitive =
    reinterpret_cast<mindspore::lite::Conv2D *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  conv_param->kernel_h_ = conv_primitive->GetKernelH();
  conv_param->kernel_w_ = conv_primitive->GetKernelW();
  conv_param->group_ = conv_primitive->GetGroup();
  conv_param->stride_h_ = conv_primitive->GetStrideH();
  conv_param->stride_w_ = conv_primitive->GetStrideW();

  auto conv2d_lite_primitive = (lite::Conv2D *)primitive;
  conv_param->pad_u_ = conv2d_lite_primitive->PadUp();
  conv_param->pad_d_ = conv2d_lite_primitive->PadDown();
  conv_param->pad_l_ = conv2d_lite_primitive->PadLeft();
  conv_param->pad_r_ = conv2d_lite_primitive->PadRight();
  conv_param->pad_h_ = conv2d_lite_primitive->PadUp();
  conv_param->pad_w_ = conv2d_lite_primitive->PadLeft();
  conv_param->dilation_h_ = conv_primitive->GetDilateH();
  conv_param->dilation_w_ = conv_primitive->GetDilateW();
  conv_param->input_channel_ = conv_primitive->GetChannelIn();
  conv_param->output_channel_ = conv_primitive->GetChannelOut();
  conv_param->group_ = conv_primitive->GetGroup();
  auto act_type = conv_primitive->GetActivationType();
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

OpParameter *PopulateConvDwParameter(const mindspore::lite::PrimitiveC *primitive) {
  ConvParameter *conv_param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "malloc ConvParameter failed.";
    return nullptr;
  }
  memset(conv_param, 0, sizeof(ConvParameter));
  conv_param->op_parameter_.type_ = primitive->Type();

  auto conv_primitive =
    reinterpret_cast<mindspore::lite::DepthwiseConv2D *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  conv_param->kernel_h_ = conv_primitive->GetKernelH();
  conv_param->kernel_w_ = conv_primitive->GetKernelW();
  conv_param->stride_h_ = conv_primitive->GetStrideH();
  conv_param->stride_w_ = conv_primitive->GetStrideW();

  auto convdw_lite_primitive = (lite::DepthwiseConv2D *)primitive;
  conv_param->pad_u_ = convdw_lite_primitive->PadUp();
  conv_param->pad_d_ = convdw_lite_primitive->PadDown();
  conv_param->pad_l_ = convdw_lite_primitive->PadLeft();
  conv_param->pad_r_ = convdw_lite_primitive->PadRight();
  conv_param->pad_h_ = convdw_lite_primitive->PadUp();
  conv_param->pad_w_ = convdw_lite_primitive->PadLeft();
  conv_param->dilation_h_ = conv_primitive->GetDilateH();
  conv_param->dilation_w_ = conv_primitive->GetDilateW();
  auto act_type = conv_primitive->GetActivationType();
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

OpParameter *PopulateDeconvDwParameter(const mindspore::lite::PrimitiveC *primitive) {
  ConvParameter *conv_param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "malloc ConvParameter failed.";
    return nullptr;
  }
  memset(conv_param, 0, sizeof(ConvParameter));
  conv_param->op_parameter_.type_ = primitive->Type();
  auto conv_primitive =
    reinterpret_cast<mindspore::lite::DeDepthwiseConv2D *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  conv_param->kernel_h_ = conv_primitive->GetKernelH();
  conv_param->kernel_w_ = conv_primitive->GetKernelW();
  conv_param->stride_h_ = conv_primitive->GetStrideH();
  conv_param->stride_w_ = conv_primitive->GetStrideW();

  auto deconvdw_lite_primitive = (mindspore::lite::DeDepthwiseConv2D *)primitive;
  conv_param->pad_u_ = deconvdw_lite_primitive->PadUp();
  conv_param->pad_d_ = deconvdw_lite_primitive->PadDown();
  conv_param->pad_l_ = deconvdw_lite_primitive->PadLeft();
  conv_param->pad_r_ = deconvdw_lite_primitive->PadRight();
  conv_param->pad_h_ = deconvdw_lite_primitive->PadUp();
  conv_param->pad_w_ = deconvdw_lite_primitive->PadLeft();
  conv_param->dilation_h_ = conv_primitive->GetDilateH();
  conv_param->dilation_w_ = conv_primitive->GetDilateW();
  auto act_type = conv_primitive->GetActivationType();
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

OpParameter *PopulateDeconvParameter(const mindspore::lite::PrimitiveC *primitive) {
  ConvParameter *conv_param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "malloc ConvParameter failed.";
    return nullptr;
  }
  memset(conv_param, 0, sizeof(ConvParameter));
  conv_param->op_parameter_.type_ = primitive->Type();
  auto conv_primitive =
    reinterpret_cast<mindspore::lite::DeConv2D *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  conv_param->kernel_h_ = conv_primitive->GetKernelH();
  conv_param->kernel_w_ = conv_primitive->GetKernelW();
  conv_param->stride_h_ = conv_primitive->GetStrideH();
  conv_param->stride_w_ = conv_primitive->GetStrideW();

  auto deconv_lite_primitive = (lite::DeConv2D *)primitive;
  conv_param->pad_u_ = deconv_lite_primitive->PadUp();
  conv_param->pad_d_ = deconv_lite_primitive->PadDown();
  conv_param->pad_l_ = deconv_lite_primitive->PadLeft();
  conv_param->pad_r_ = deconv_lite_primitive->PadRight();
  conv_param->pad_h_ = deconv_lite_primitive->PadH();
  conv_param->pad_w_ = deconv_lite_primitive->PadW();
  conv_param->dilation_h_ = conv_primitive->GetDilateH();
  conv_param->dilation_w_ = conv_primitive->GetDilateW();
  auto act_type = conv_primitive->GetActivationType();
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

OpParameter *PopulateSoftmaxParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto softmax_primitive =
    reinterpret_cast<mindspore::lite::SoftMax *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  SoftmaxParameter *softmax_param = reinterpret_cast<SoftmaxParameter *>(malloc(sizeof(SoftmaxParameter)));
  if (softmax_param == nullptr) {
    MS_LOG(ERROR) << "malloc SoftmaxParameter failed.";
    return nullptr;
  }
  memset(softmax_param, 0, sizeof(SoftmaxParameter));
  softmax_param->op_parameter_.type_ = primitive->Type();
  softmax_param->axis_ = softmax_primitive->GetAxis();
  return reinterpret_cast<OpParameter *>(softmax_param);
}

OpParameter *PopulateReduceParameter(const mindspore::lite::PrimitiveC *primitive) {
  ReduceParameter *reduce_param = reinterpret_cast<ReduceParameter *>(malloc(sizeof(ReduceParameter)));
  if (reduce_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReduceParameter failed.";
    return nullptr;
  }
  memset(reduce_param, 0, sizeof(ReduceParameter));
  reduce_param->op_parameter_.type_ = primitive->Type();
  auto reduce = reinterpret_cast<mindspore::lite::Reduce *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  reduce_param->keep_dims_ = reduce->GetKeepDims();
  auto axisVector = reduce->GetAxes();
  if (axisVector.size() > REDUCE_MAX_AXES_NUM) {
    MS_LOG(ERROR) << "Reduce axes size " << axisVector.size() << " exceed limit " << REDUCE_MAX_AXES_NUM;
    free(reduce_param);
    return nullptr;
  }
  reduce_param->num_axes_ = static_cast<int>(axisVector.size());
  int i = 0;
  for (auto iter = axisVector.begin(); iter != axisVector.end(); iter++) {
    reduce_param->axes_[i++] = *iter;
  }
  reduce_param->mode_ = static_cast<int>(reduce->GetMode());
  return reinterpret_cast<OpParameter *>(reduce_param);
}

OpParameter *PopulateMeanParameter(const mindspore::lite::PrimitiveC *primitive) {
  ReduceParameter *mean_param = reinterpret_cast<ReduceParameter *>(malloc(sizeof(ReduceParameter)));
  if (mean_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReduceParameter failed.";
    return nullptr;
  }
  memset(mean_param, 0, sizeof(ReduceParameter));
  mean_param->op_parameter_.type_ = primitive->Type();
  auto mean = reinterpret_cast<mindspore::lite::Mean *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  mean_param->keep_dims_ = mean->GetKeepDims();
  auto axisVector = mean->GetAxis();
  if (axisVector.size() > REDUCE_MAX_AXES_NUM) {
    MS_LOG(ERROR) << "Reduce axes size " << axisVector.size() << " exceed limit " << REDUCE_MAX_AXES_NUM;
    free(mean_param);
    return nullptr;
  }
  mean_param->num_axes_ = static_cast<int>(axisVector.size());
  int i = 0;
  for (auto iter = axisVector.begin(); iter != axisVector.end(); iter++) {
    mean_param->axes_[i++] = *iter;
  }
  mean_param->mode_ = static_cast<int>(schema::ReduceMode_ReduceMean);
  return reinterpret_cast<OpParameter *>(mean_param);
}

OpParameter *PopulatePadParameter(const mindspore::lite::PrimitiveC *primitive) {
  PadParameter *pad_param = reinterpret_cast<PadParameter *>(malloc(sizeof(PadParameter)));
  if (pad_param == nullptr) {
    MS_LOG(ERROR) << "malloc PadParameter failed.";
    return nullptr;
  }
  memset(pad_param, 0, sizeof(PadParameter));
  pad_param->op_parameter_.type_ = primitive->Type();
  auto pad_node = reinterpret_cast<mindspore::lite::Pad *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  pad_param->pad_mode_ = pad_node->GetPaddingMode();
  if (pad_param->pad_mode_ == schema::PaddingMode_CONSTANT) {
    pad_param->constant_value_ = pad_node->GetConstantValue();
  } else {
    MS_LOG(ERROR) << "Invalid padding mode: " << pad_param->pad_mode_;
    free(pad_param);
    return nullptr;
  }

  auto size = pad_node->GetPaddings().size();
  if (size > MAX_PAD_SIZE) {
    MS_LOG(ERROR) << "Invalid padding size: " << size;
    free(pad_param);
    return nullptr;
  }

  for (size_t i = 0; i < size; i++) {
    pad_param->paddings_[MAX_PAD_SIZE - size + i] = pad_node->GetPaddings()[i];
  }
  return reinterpret_cast<OpParameter *>(pad_param);
}

OpParameter *PopulateActivationParameter(const mindspore::lite::PrimitiveC *primitive) {
  ActivationParameter *act_param = reinterpret_cast<ActivationParameter *>(malloc(sizeof(ActivationParameter)));
  if (act_param == nullptr) {
    MS_LOG(ERROR) << "malloc ActivationParameter failed.";
    return nullptr;
  }
  memset(act_param, 0, sizeof(ActivationParameter));
  auto activation =
    reinterpret_cast<mindspore::lite::Activation *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  act_param->type_ = static_cast<int>(activation->GetType());
  act_param->alpha_ = activation->GetAlpha();
  return reinterpret_cast<OpParameter *>(act_param);
}

OpParameter *PopulateFusedBatchNorm(const mindspore::lite::PrimitiveC *primitive) {
  BatchNormParameter *batch_norm_param = reinterpret_cast<BatchNormParameter *>(malloc(sizeof(BatchNormParameter)));
  if (batch_norm_param == nullptr) {
    MS_LOG(ERROR) << "malloc BatchNormParameter failed.";
    return nullptr;
  }
  memset(batch_norm_param, 0, sizeof(BatchNormParameter));
  batch_norm_param->op_parameter_.type_ = primitive->Type();
  auto param =
    reinterpret_cast<mindspore::lite::FusedBatchNorm *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  batch_norm_param->epsilon_ = param->GetEpsilon();
  batch_norm_param->fused_ = true;
  return reinterpret_cast<OpParameter *>(batch_norm_param);
}

OpParameter *PopulateArithmetic(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *arithmetic_param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(arithmetic_param, 0, sizeof(ArithmeticParameter));
  arithmetic_param->op_parameter_.type_ = primitive->Type();
  arithmetic_param->broadcasting_ = ((lite::Arithmetic *)primitive)->Broadcasting();
  arithmetic_param->ndim_ = ((lite::Arithmetic *)primitive)->NDims();
  switch (primitive->Type()) {
    case schema::PrimitiveType_Add:
      arithmetic_param->activation_type_ =
        reinterpret_cast<mindspore::lite::Add *>(const_cast<mindspore::lite::PrimitiveC *>(primitive))
          ->GetActivationType();
      break;
    case schema::PrimitiveType_Sub:
      arithmetic_param->activation_type_ =
        reinterpret_cast<mindspore::lite::Sub *>(const_cast<mindspore::lite::PrimitiveC *>(primitive))
          ->GetActivationType();
      break;
    case schema::PrimitiveType_Mul:
      arithmetic_param->activation_type_ =
        reinterpret_cast<mindspore::lite::Mul *>(const_cast<mindspore::lite::PrimitiveC *>(primitive))
          ->GetActivationType();
      break;
    case schema::PrimitiveType_Div:
      arithmetic_param->activation_type_ =
        reinterpret_cast<mindspore::lite::Div *>(const_cast<mindspore::lite::PrimitiveC *>(primitive))
          ->GetActivationType();
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

OpParameter *PopulateEltwiseParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *arithmetic_param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(arithmetic_param, 0, sizeof(ArithmeticParameter));
  auto eltwise = reinterpret_cast<mindspore::lite::Eltwise *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  switch (eltwise->GetMode()) {
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
      free(arithmetic_param);
      return nullptr;
  }
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

OpParameter *PopulateArithmeticSelf(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticSelfParameter *arithmetic_self_param =
      reinterpret_cast<ArithmeticSelfParameter *>(malloc(sizeof(ArithmeticSelfParameter)));
  if (arithmetic_self_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticSelfParameter failed.";
    return nullptr;
  }
  memset(arithmetic_self_param, 0, sizeof(ArithmeticSelfParameter));
  arithmetic_self_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(arithmetic_self_param);
}

OpParameter *PopulatePowerParameter(const mindspore::lite::PrimitiveC *primitive) {
  PowerParameter *power_param = reinterpret_cast<PowerParameter *>(malloc(sizeof(PowerParameter)));
  if (power_param == nullptr) {
    MS_LOG(ERROR) << "malloc PowerParameter failed.";
    return nullptr;
  }
  memset(power_param, 0, sizeof(PowerParameter));
  power_param->op_parameter_.type_ = primitive->Type();
  auto power = reinterpret_cast<mindspore::lite::Power *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  power_param->power_ = power->GetPower();
  power_param->scale_ = power->GetScale();
  power_param->shift_ = power->GetShift();
  return reinterpret_cast<OpParameter *>(power_param);
}

OpParameter *PopulateArgMaxParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArgMinMaxParameter *arg_param = reinterpret_cast<ArgMinMaxParameter *>(malloc(sizeof(ArgMinMaxParameter)));
  if (arg_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArgMinMaxParameter failed.";
    return nullptr;
  }
  memset(arg_param, 0, sizeof(ArgMinMaxParameter));
  arg_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::ArgMax *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  arg_param->axis_ = param->GetAxis();
  arg_param->topk_ = param->GetTopK();
  arg_param->axis_type_ = param->GetAxisType();
  arg_param->out_value_ = param->GetOutMaxValue();
  arg_param->keep_dims_ = param->GetKeepDims();
  return reinterpret_cast<OpParameter *>(arg_param);
}

OpParameter *PopulateArgMinParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArgMinMaxParameter *arg_param = reinterpret_cast<ArgMinMaxParameter *>(malloc(sizeof(ArgMinMaxParameter)));
  if (arg_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArgMinMaxParameter failed.";
    return nullptr;
  }
  memset(arg_param, 0, sizeof(ArgMinMaxParameter));
  arg_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::ArgMin *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  arg_param->axis_ = param->GetAxis();
  arg_param->topk_ = param->GetTopK();
  arg_param->axis_type_ = param->GetAxisType();
  arg_param->out_value_ = param->GetOutMaxValue();
  arg_param->keep_dims_ = param->GetKeepDims();
  return reinterpret_cast<OpParameter *>(arg_param);
}

OpParameter *PopulateCastParameter(const mindspore::lite::PrimitiveC *primitive) {
  CastParameter *cast_param = reinterpret_cast<CastParameter *>(malloc(sizeof(CastParameter)));
  if (cast_param == nullptr) {
    MS_LOG(ERROR) << "malloc CastParameter failed.";
    return nullptr;
  }
  memset(cast_param, 0, sizeof(CastParameter));
  cast_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Cast *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  cast_param->src_type_ = param->GetSrcT();
  cast_param->dst_type_ = param->GetDstT();
  return reinterpret_cast<OpParameter *>(cast_param);
}

OpParameter *PopulateLocalResponseNormParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto local_response_norm_attr = reinterpret_cast<mindspore::lite::LocalResponseNormalization *>(
    const_cast<mindspore::lite::PrimitiveC *>(primitive));
  LocalResponseNormParameter *lrn_param =
      reinterpret_cast<LocalResponseNormParameter *>(malloc(sizeof(LocalResponseNormParameter)));
  if (lrn_param == nullptr) {
    MS_LOG(ERROR) << "malloc LocalResponseNormParameter failed.";
    return nullptr;
  }
  memset(lrn_param, 0, sizeof(LocalResponseNormParameter));
  lrn_param->op_parameter_.type_ = primitive->Type();
  lrn_param->depth_radius_ = local_response_norm_attr->GetDepthRadius();
  lrn_param->bias_ = local_response_norm_attr->GetBias();
  lrn_param->alpha_ = local_response_norm_attr->GetAlpha();
  lrn_param->beta_ = local_response_norm_attr->GetBeta();
  return reinterpret_cast<OpParameter *>(lrn_param);
}

OpParameter *PopulateRangeParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto range_attr = reinterpret_cast<mindspore::lite::Range *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  RangeParameter *range_param = reinterpret_cast<RangeParameter *>(malloc(sizeof(RangeParameter)));
  if (range_param == nullptr) {
    MS_LOG(ERROR) << "malloc RangeParameter failed.";
    return nullptr;
  }
  memset(range_param, 0, sizeof(RangeParameter));
  range_param->op_parameter_.type_ = primitive->Type();
  range_param->start_ = range_attr->GetStart();
  range_param->limit_ = range_attr->GetLimit();
  range_param->delta_ = range_attr->GetDelta();
  range_param->dType_ = range_attr->GetDType();
  return reinterpret_cast<OpParameter *>(range_param);
}

OpParameter *PopulateConcatParameter(const mindspore::lite::PrimitiveC *primitive) {
  ConcatParameter *concat_param = reinterpret_cast<ConcatParameter *>(malloc(sizeof(ConcatParameter)));
  if (concat_param == nullptr) {
    MS_LOG(ERROR) << "malloc ConcatParameter failed.";
    return nullptr;
  }
  memset(concat_param, 0, sizeof(ConcatParameter));
  concat_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Concat *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  concat_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(concat_param);
}

OpParameter *PopulateTileParameter(const mindspore::lite::PrimitiveC *primitive) {
  TileParameter *tile_param = reinterpret_cast<TileParameter *>(malloc(sizeof(TileParameter)));
  if (tile_param == nullptr) {
    MS_LOG(ERROR) << "malloc TileParameter failed.";
    return nullptr;
  }
  memset(tile_param, 0, sizeof(TileParameter));
  tile_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Tile *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto multiples = param->GetMultiples();
  tile_param->in_dim_ = multiples.size();
  for (int i = 0; i < tile_param->in_dim_; ++i) {
    tile_param->multiples_[i] = multiples[i];
  }
  return reinterpret_cast<OpParameter *>(tile_param);
}

OpParameter *PopulateTopKParameter(const mindspore::lite::PrimitiveC *primitive) {
  TopkParameter *topk_param = reinterpret_cast<TopkParameter *>(malloc(sizeof(TopkParameter)));
  if (topk_param == nullptr) {
    MS_LOG(ERROR) << "malloc TopkParameter failed.";
    return nullptr;
  }
  memset(topk_param, 0, sizeof(TopkParameter));
  topk_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::TopK *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  topk_param->k_ = param->GetK();
  topk_param->sorted_ = param->GetSorted();
  return reinterpret_cast<OpParameter *>(topk_param);
}

OpParameter *PopulateNhwc2NchwParameter(const mindspore::lite::PrimitiveC *primitive) {
  OpParameter *parameter = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "malloc OpParameter failed.";
    return nullptr;
  }
  memset(parameter, 0, sizeof(OpParameter));
  parameter->type_ = primitive->Type();
  return parameter;
}

OpParameter *PopulateNchw2NhwcParameter(const mindspore::lite::PrimitiveC *primitive) {
  OpParameter *parameter = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "malloc OpParameter failed.";
    return nullptr;
  }
  memset(parameter, 0, sizeof(OpParameter));
  parameter->type_ = primitive->Type();
  return parameter;
}

OpParameter *PopulateTransposeParameter(const mindspore::lite::PrimitiveC *primitive) {
  TransposeParameter *transpose_param = reinterpret_cast<TransposeParameter *>(malloc(sizeof(TransposeParameter)));
  if (transpose_param == nullptr) {
    MS_LOG(ERROR) << "malloc TransposeParameter failed.";
    return nullptr;
  }
  memset(transpose_param, 0, sizeof(TransposeParameter));
  auto param = reinterpret_cast<mindspore::lite::Transpose *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  transpose_param->op_parameter_.type_ = primitive->Type();
  auto perm_vector_ = param->GetPerm();
  int i = 0;
  for (auto iter = perm_vector_.begin(); iter != perm_vector_.end(); iter++) {
    transpose_param->perm_[i++] = *iter;
  }
  transpose_param->num_axes_ = i;
  transpose_param->conjugate_ = param->GetConjugate();
  return reinterpret_cast<OpParameter *>(transpose_param);
}

OpParameter *PopulateSplitParameter(const mindspore::lite::PrimitiveC *primitive) {
  SplitParameter *split_param = reinterpret_cast<SplitParameter *>(malloc(sizeof(SplitParameter)));
  if (split_param == nullptr) {
    MS_LOG(ERROR) << "malloc SplitParameter failed.";
    return nullptr;
  }
  memset(split_param, 0, sizeof(SplitParameter));
  auto param = reinterpret_cast<mindspore::lite::Split *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  split_param->op_parameter_.type_ = primitive->Type();
  split_param->num_split_ = param->GetNumberSplit();
  auto split_sizes_vector_ = param->GetSizeSplits();
  int i = 0;
  for (auto iter = split_sizes_vector_.begin(); iter != split_sizes_vector_.end(); iter++) {
    split_param->split_sizes_[i++] = *iter;
  }
  split_param->split_dim_ = param->GetSplitDim();
  split_param->num_split_ = param->GetNumberSplit();
  return reinterpret_cast<OpParameter *>(split_param);
}

OpParameter *PopulateSqueezeParameter(const mindspore::lite::PrimitiveC *primitive) {
  SqueezeParameter *squeeze_param = reinterpret_cast<SqueezeParameter *>(malloc(sizeof(SqueezeParameter)));
  if (squeeze_param == nullptr) {
    MS_LOG(ERROR) << "malloc SqueezeParameter failed.";
    return nullptr;
  }
  memset(squeeze_param, 0, sizeof(SqueezeParameter));
  squeeze_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(squeeze_param);
}

OpParameter *PopulateScaleParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "input primitive is nullptr";
    return nullptr;
  }
  ScaleParameter *scale_param = reinterpret_cast<ScaleParameter *>(malloc(sizeof(ScaleParameter)));
  if (scale_param == nullptr) {
    MS_LOG(ERROR) << "malloc ScaleParameter failed.";
    return nullptr;
  }
  memset(scale_param, 0, sizeof(ScaleParameter));
  scale_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Scale *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  scale_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(scale_param);
}

OpParameter *PopulateGatherParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto gather_attr = reinterpret_cast<mindspore::lite::Gather *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  GatherParameter *gather_param = reinterpret_cast<GatherParameter *>(malloc(sizeof(GatherParameter)));
  if (gather_param == nullptr) {
    MS_LOG(ERROR) << "malloc GatherParameter failed.";
    return nullptr;
  }
  memset(gather_param, 0, sizeof(GatherParameter));
  gather_param->op_parameter_.type_ = primitive->Type();
  gather_param->axis_ = gather_attr->GetAxis();
  gather_param->batchDims_ = gather_attr->GetBatchDims();
  return reinterpret_cast<OpParameter *>(gather_param);
}

OpParameter *PopulateGatherNdParameter(const mindspore::lite::PrimitiveC *primitive) {
  GatherNdParameter *gather_nd_param = reinterpret_cast<GatherNdParameter *>(malloc(sizeof(GatherNdParameter)));
  if (gather_nd_param == nullptr) {
    MS_LOG(ERROR) << "malloc GatherNdParameter failed.";
    return nullptr;
  }
  memset(gather_nd_param, 0, sizeof(GatherNdParameter));
  gather_nd_param->op_parameter_.type_ = primitive->Type();
  auto gatherNd_attr =
    reinterpret_cast<mindspore::lite::GatherNd *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  gather_nd_param->batchDims_ = gatherNd_attr->GetBatchDims();
  return reinterpret_cast<OpParameter *>(gather_nd_param);
}

OpParameter *PopulateScatterNDParameter(const mindspore::lite::PrimitiveC *primitive) {
  ScatterNDParameter *scatter_nd_param = reinterpret_cast<ScatterNDParameter *>(malloc(sizeof(ScatterNDParameter)));
  if (scatter_nd_param == nullptr) {
    MS_LOG(ERROR) << "malloc ScatterNDParameter failed.";
    return nullptr;
  }
  memset(scatter_nd_param, 0, sizeof(ScatterNDParameter));
  scatter_nd_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(scatter_nd_param);
}

OpParameter *PopulateSliceParameter(const mindspore::lite::PrimitiveC *primitive) {
  SliceParameter *slice_param = reinterpret_cast<SliceParameter *>(malloc(sizeof(SliceParameter)));
  if (slice_param == nullptr) {
    MS_LOG(ERROR) << "malloc SliceParameter failed.";
    return nullptr;
  }
  memset(slice_param, 0, sizeof(SliceParameter));
  auto param = reinterpret_cast<mindspore::lite::Slice *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  slice_param->op_parameter_.type_ = primitive->Type();
  auto param_begin = param->GetBegin();
  auto param_size = param->GetSize();
  if (param_begin.size() != param_size.size()) {
    free(slice_param);
    return nullptr;
  }
  slice_param->param_length_ = static_cast<int32_t>(param_begin.size());
  for (int32_t i = 0; i < slice_param->param_length_; ++i) {
    slice_param->begin_[i] = param_begin[i];
    slice_param->size_[i] = param_size[i];
  }
  return reinterpret_cast<OpParameter *>(slice_param);
}

OpParameter *PopulateBroadcastToParameter(const mindspore::lite::PrimitiveC *primitive) {
  BroadcastToParameter *broadcast_param =
      reinterpret_cast<BroadcastToParameter *>(malloc(sizeof(BroadcastToParameter)));
  if (broadcast_param == nullptr) {
    MS_LOG(ERROR) << "malloc BroadcastToParameter failed.";
    return nullptr;
  }
  memset(broadcast_param, 0, sizeof(BroadcastToParameter));
  auto param = reinterpret_cast<mindspore::lite::BroadcastTo *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  broadcast_param->op_parameter_.type_ = primitive->Type();
  auto dst_shape = param->GetDstShape();
  broadcast_param->shape_size_ = dst_shape.size();
  for (size_t i = 0; i < broadcast_param->shape_size_; ++i) {
    broadcast_param->shape_[i] = dst_shape[i];
  }
  return reinterpret_cast<OpParameter *>(broadcast_param);
}

OpParameter *PopulateReshapeParameter(const mindspore::lite::PrimitiveC *primitive) {
  ReshapeParameter *reshape_param = reinterpret_cast<ReshapeParameter *>(malloc(sizeof(ReshapeParameter)));
  if (reshape_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReshapeParameter failed.";
    return nullptr;
  }
  memset(reshape_param, 0, sizeof(ReshapeParameter));
  reshape_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(reshape_param);
}

OpParameter *PopulateShapeParameter(const mindspore::lite::PrimitiveC *primitive) {
  ShapeParameter *shape_param = reinterpret_cast<ShapeParameter *>(malloc(sizeof(ShapeParameter)));
  if (shape_param == nullptr) {
    MS_LOG(ERROR) << "malloc ShapeParameter failed.";
    return nullptr;
  }
  memset(shape_param, 0, sizeof(ShapeParameter));
  shape_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(shape_param);
}

OpParameter *PopulateConstantOfShapeParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto attr =
    reinterpret_cast<mindspore::lite::ConstantOfShape *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  ConstantOfShapeParameter *param =
      reinterpret_cast<ConstantOfShapeParameter *>(malloc(sizeof(ConstantOfShapeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ConstantOfShapeParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ConstantOfShapeParameter));
  param->op_parameter_.type_ = primitive->Type();
  param->value_ = attr->GetValue();
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateReverseParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto reverse_attr =
    reinterpret_cast<mindspore::lite::Reverse *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  ReverseParameter *reverse_param = reinterpret_cast<ReverseParameter *>(malloc(sizeof(ReverseParameter)));
  if (reverse_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReverseParameter failed.";
    return nullptr;
  }
  memset(reverse_param, 0, sizeof(ReverseParameter));
  reverse_param->op_parameter_.type_ = primitive->Type();
  auto flatAxis = reverse_attr->GetAxis();
  reverse_param->num_axis_ = flatAxis.size();
  int i = 0;
  for (auto iter = flatAxis.begin(); iter != flatAxis.end(); iter++) {
    reverse_param->axis_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(reverse_param);
}

OpParameter *PopulateUnsqueezeParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto unsqueeze_attr =
    reinterpret_cast<mindspore::lite::Unsqueeze *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  UnsqueezeParameter *unsqueeze_param = reinterpret_cast<UnsqueezeParameter *>(malloc(sizeof(UnsqueezeParameter)));
  if (unsqueeze_param == nullptr) {
    MS_LOG(ERROR) << "malloc UnsqueezeParameter failed.";
    return nullptr;
  }
  memset(unsqueeze_param, 0, sizeof(UnsqueezeParameter));
  unsqueeze_param->op_parameter_.type_ = primitive->Type();
  auto flatAxis = unsqueeze_attr->GetAxis();
  unsqueeze_param->num_dim_ = flatAxis.size();
  int i = 0;
  for (auto iter = flatAxis.begin(); iter != flatAxis.end(); iter++) {
    unsqueeze_param->dims_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(unsqueeze_param);
}

OpParameter *PopulateStackParameter(const mindspore::lite::PrimitiveC *primitive) {
  StackParameter *stack_param = reinterpret_cast<StackParameter *>(malloc(sizeof(StackParameter)));
  if (stack_param == nullptr) {
    MS_LOG(ERROR) << "malloc StackParameter failed.";
    return nullptr;
  }
  memset(stack_param, 0, sizeof(StackParameter));
  auto param = reinterpret_cast<mindspore::lite::Stack *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  stack_param->op_parameter_.type_ = primitive->Type();
  stack_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(stack_param);
}

OpParameter *PopulateUnstackParameter(const mindspore::lite::PrimitiveC *primitive) {
  UnstackParameter *unstack_param = reinterpret_cast<UnstackParameter *>(malloc(sizeof(UnstackParameter)));
  if (unstack_param == nullptr) {
    MS_LOG(ERROR) << "malloc UnstackParameter failed.";
    return nullptr;
  }
  memset(unstack_param, 0, sizeof(UnstackParameter));
  auto param = reinterpret_cast<mindspore::lite::Unstack *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  unstack_param->op_parameter_.type_ = primitive->Type();
  unstack_param->num_ = param->GetNum();
  unstack_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(unstack_param);
}

OpParameter *PopulateReverseSequenceParameter(const mindspore::lite::PrimitiveC *primitive) {
  ReverseSequenceParameter *reverse_sequence_param =
      reinterpret_cast<ReverseSequenceParameter *>(malloc(sizeof(ReverseSequenceParameter)));
  if (reverse_sequence_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReverseSequenceParameter failed.";
    return nullptr;
  }
  memset(reverse_sequence_param, 0, sizeof(ReverseSequenceParameter));
  auto param =
    reinterpret_cast<mindspore::lite::ReverseSequence *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  reverse_sequence_param->op_parameter_.type_ = primitive->Type();
  reverse_sequence_param->seq_axis_ = param->GetSeqAxis();
  reverse_sequence_param->batch_axis_ = param->GetBatchAxis();
  return reinterpret_cast<OpParameter *>(reverse_sequence_param);
}

OpParameter *PopulateUniqueParameter(const mindspore::lite::PrimitiveC *primitive) {
  UniqueParameter *unique_param = reinterpret_cast<UniqueParameter *>(malloc(sizeof(UniqueParameter)));
  if (unique_param == nullptr) {
    MS_LOG(ERROR) << "malloc UniqueParameter failed.";
    return nullptr;
  }
  memset(unique_param, 0, sizeof(UniqueParameter));
  unique_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(unique_param);
}

OpParameter *PopulateDepthToSpaceParameter(const mindspore::lite::PrimitiveC *primitive) {
  DepthToSpaceParameter *depth_space_param =
      reinterpret_cast<DepthToSpaceParameter *>(malloc(sizeof(DepthToSpaceParameter)));
  if (depth_space_param == nullptr) {
    MS_LOG(ERROR) << "malloc DepthToSpaceParameter failed.";
    return nullptr;
  }
  memset(depth_space_param, 0, sizeof(DepthToSpaceParameter));
  auto param = reinterpret_cast<mindspore::lite::DepthToSpace *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  depth_space_param->op_parameter_.type_ = primitive->Type();
  depth_space_param->block_size_ = param->GetBlockSize();
  return reinterpret_cast<OpParameter *>(depth_space_param);
}

OpParameter *PopulateSpaceToDepthParameter(const mindspore::lite::PrimitiveC *primitive) {
  SpaceToDepthParameter *space_depth_param =
      reinterpret_cast<SpaceToDepthParameter *>(malloc(sizeof(SpaceToDepthParameter)));
  if (space_depth_param == nullptr) {
    MS_LOG(ERROR) << "malloc SpaceToDepthParameter failed.";
    return nullptr;
  }
  memset(space_depth_param, 0, sizeof(SpaceToDepthParameter));
  space_depth_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::SpaceToDepth *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  space_depth_param->op_parameter_.type_ = primitive->Type();
  space_depth_param->block_size_ = param->GetBlockSize();
  if (param->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "Currently only NHWC format is supported.";
    free(space_depth_param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(space_depth_param);
}

OpParameter *PopulateSpaceToBatchParameter(const mindspore::lite::PrimitiveC *primitive) {
  SpaceToBatchParameter *space_batch_param =
      reinterpret_cast<SpaceToBatchParameter *>(malloc(sizeof(SpaceToBatchParameter)));
  if (space_batch_param == nullptr) {
    MS_LOG(ERROR) << "malloc SpaceToBatchParameter failed.";
    return nullptr;
  }
  memset(space_batch_param, 0, sizeof(SpaceToBatchParameter));
  space_batch_param->op_parameter_.type_ = primitive->Type();
  space_batch_param->op_parameter_.type_ = primitive->Type();
  auto block_sizes = ((mindspore::lite::SpaceToBatch *)primitive)->BlockSizes();
  (void)memcpy(space_batch_param->block_sizes_, (block_sizes.data()), block_sizes.size() * sizeof(int));
  auto paddings = ((mindspore::lite::SpaceToBatch *)primitive)->Paddings();
  (void)memcpy(space_batch_param->paddings_, (paddings.data()), paddings.size() * sizeof(int));
  auto in_shape = ((mindspore::lite::SpaceToBatch *)primitive)->InShape();
  (void)memcpy(space_batch_param->in_shape_, (in_shape.data()), in_shape.size() * sizeof(int));
  auto padded_in_shape = ((mindspore::lite::SpaceToBatch *)primitive)->PaddedInShape();
  (void)memcpy(space_batch_param->padded_in_shape_, (padded_in_shape.data()), padded_in_shape.size() * sizeof(int));
  return reinterpret_cast<OpParameter *>(space_batch_param);
}

OpParameter *PopulateResizeParameter(const mindspore::lite::PrimitiveC *primitive) {
  ResizeParameter *resize_param = reinterpret_cast<ResizeParameter *>(malloc(sizeof(ResizeParameter)));
  if (resize_param == nullptr) {
    MS_LOG(ERROR) << "malloc ResizeParameter failed.";
    return nullptr;
  }
  memset(resize_param, 0, sizeof(ResizeParameter));
  resize_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Resize *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  resize_param->method_ = static_cast<int>(param->GetMethod());
  resize_param->new_height_ = param->GetNewHeight();
  resize_param->new_width_ = param->GetNewWidth();
  resize_param->align_corners_ = param->GetAlignCorners();
  resize_param->preserve_aspect_ratio_ = param->GetPreserveAspectRatio();
  return reinterpret_cast<OpParameter *>(resize_param);
}

OpParameter *PopulateBatchToSpaceParameter(const mindspore::lite::PrimitiveC *primitive) {
  BatchToSpaceParameter *batch_space_param =
      reinterpret_cast<BatchToSpaceParameter *>(malloc(sizeof(BatchToSpaceParameter)));
  if (batch_space_param == nullptr) {
    MS_LOG(ERROR) << "malloc BatchToSpaceParameter failed.";
    return nullptr;
  }
  memset(batch_space_param, 0, sizeof(BatchToSpaceParameter));
  batch_space_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::BatchToSpace *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto block_shape = param->GetBlockShape();
  if (block_shape.size() != BATCH_TO_SPACE_BLOCK_SHAPE_SIZE) {
    MS_LOG(ERROR) << "batch_to_space blockShape size should be " << BATCH_TO_SPACE_BLOCK_SHAPE_SIZE;
    free(batch_space_param);
    return nullptr;
  }

  auto crops = param->GetCrops();
  if (crops.size() != BATCH_TO_SPACE_CROPS_SIZE) {
    MS_LOG(ERROR) << "batch_to_space crops size should be " << BATCH_TO_SPACE_CROPS_SIZE;
    free(batch_space_param);
    return nullptr;
  }

  for (int i = 0; i < BATCH_TO_SPACE_BLOCK_SHAPE_SIZE; ++i) {
    batch_space_param->block_shape_[i] = block_shape[i];
  }

  for (int i = 0; i < BATCH_TO_SPACE_CROPS_SIZE; ++i) {
    batch_space_param->crops_[i] = crops[i];
  }
  return reinterpret_cast<OpParameter *>(batch_space_param);
}

OpParameter *PopulateCropParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto param = reinterpret_cast<mindspore::lite::Crop *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto param_offset = param->GetOffsets();
  if (param_offset.size() > CROP_OFFSET_MAX_SIZE) {
    MS_LOG(ERROR) << "crop_param offset size(" << param_offset.size() << ") should <= " << CROP_OFFSET_MAX_SIZE;
    return nullptr;
  }
  CropParameter *crop_param = reinterpret_cast<CropParameter *>(malloc(sizeof(CropParameter)));
  if (crop_param == nullptr) {
    MS_LOG(ERROR) << "malloc CropParameter failed.";
    return nullptr;
  }
  memset(crop_param, 0, sizeof(CropParameter));
  crop_param->op_parameter_.type_ = primitive->Type();
  crop_param->axis_ = param->GetAxis();
  crop_param->offset_size_ = param_offset.size();
  for (size_t i = 0; i < param_offset.size(); ++i) {
    crop_param->offset_[i] = param_offset[i];
  }
  return reinterpret_cast<OpParameter *>(crop_param);
}

OpParameter *PopulateOneHotParameter(const mindspore::lite::PrimitiveC *primitive) {
  OneHotParameter *one_hot_param = reinterpret_cast<OneHotParameter *>(malloc(sizeof(OneHotParameter)));
  if (one_hot_param == nullptr) {
    MS_LOG(ERROR) << "malloc OneHotParameter failed.";
    return nullptr;
  }
  memset(one_hot_param, 0, sizeof(OneHotParameter));
  one_hot_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::OneHot *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  if (param == nullptr) {
    free(one_hot_param);
    MS_LOG(ERROR) << "get OneHot param nullptr.";
    return nullptr;
  }
  one_hot_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(one_hot_param);
}

OpParameter *PopulateFlattenParameter(const mindspore::lite::PrimitiveC *primitive) {
  FlattenParameter *flatten_param = reinterpret_cast<FlattenParameter *>(malloc(sizeof(FlattenParameter)));
  if (flatten_param == nullptr) {
    MS_LOG(ERROR) << "malloc FlattenParameter failed.";
    return nullptr;
  }
  memset(flatten_param, 0, sizeof(FlattenParameter));
  flatten_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(flatten_param);
}

OpParameter *PopulateQuantDTypeCastParameter(const mindspore::lite::PrimitiveC *primitive) {
  QuantDTypeCastParameter *parameter =
      reinterpret_cast<QuantDTypeCastParameter *>(malloc(sizeof(QuantDTypeCastParameter)));
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "malloc QuantDTypeCastParameter failed.";
    return nullptr;
  }
  memset(parameter, 0, sizeof(QuantDTypeCastParameter));
  parameter->op_parameter_.type_ = primitive->Type();
  auto quant_dtype_cast_param =
    reinterpret_cast<mindspore::lite::QuantDTypeCast *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  parameter->srcT = quant_dtype_cast_param->GetSrcT();
  parameter->dstT = quant_dtype_cast_param->GetDstT();
  return reinterpret_cast<OpParameter *>(parameter);
}

OpParameter *PopulateStridedSliceParameter(const mindspore::lite::PrimitiveC *primitive) {
  StridedSliceParameter *strided_slice_param =
      reinterpret_cast<StridedSliceParameter *>(malloc(sizeof(StridedSliceParameter)));
  if (strided_slice_param == nullptr) {
    MS_LOG(ERROR) << "malloc StridedSliceParameter failed.";
    return nullptr;
  }
  memset(strided_slice_param, 0, sizeof(StridedSliceParameter));
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

OpParameter *PopulateAddNParameter(const mindspore::lite::PrimitiveC *primitive) {
  OpParameter *addn_param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (addn_param == nullptr) {
    MS_LOG(ERROR) << "malloc OpParameter failed.";
    return nullptr;
  }
  memset(addn_param, 0, sizeof(OpParameter));
  addn_param->type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(addn_param);
}

OpParameter *PopulatePriorBoxParameter(const mindspore::lite::PrimitiveC *primitive) {
  PriorBoxParameter *prior_box_param = reinterpret_cast<PriorBoxParameter *>(malloc(sizeof(PriorBoxParameter)));
  if (prior_box_param == nullptr) {
    MS_LOG(ERROR) << "malloc PriorBoxParameter failed.";
    return nullptr;
  }
  memset(prior_box_param, 0, sizeof(PriorBoxParameter));
  prior_box_param->op_parameter_.type_ = primitive->Type();
  auto prior_box_attr =
    reinterpret_cast<mindspore::lite::PriorBox *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));

  if (prior_box_attr->GetMinSizes().size() > PRIOR_BOX_MAX_NUM) {
    MS_LOG(ERROR) << "PriorBox min_sizes size exceeds max num " << PRIOR_BOX_MAX_NUM << ", got "
                  << prior_box_attr->GetMinSizes();
    free(prior_box_param);
    return nullptr;
  }
  prior_box_param->min_sizes_size = prior_box_attr->GetMinSizes().size();
  if (prior_box_attr->GetMaxSizes().size() > PRIOR_BOX_MAX_NUM) {
    MS_LOG(ERROR) << "PriorBox max_sizes size exceeds max num " << PRIOR_BOX_MAX_NUM << ", got "
                  << prior_box_attr->GetMaxSizes();
    free(prior_box_param);
    return nullptr;
  }
  prior_box_param->max_sizes_size = prior_box_attr->GetMaxSizes().size();
  (void)memcpy(prior_box_param->max_sizes, prior_box_attr->GetMaxSizes().data(),
               prior_box_attr->GetMaxSizes().size() * sizeof(int32_t));
  (void)memcpy(prior_box_param->min_sizes, prior_box_attr->GetMinSizes().data(),
               prior_box_attr->GetMinSizes().size() * sizeof(int32_t));

  if (prior_box_attr->GetAspectRatios().size() > PRIOR_BOX_MAX_NUM) {
    MS_LOG(ERROR) << "PriorBox aspect_ratios size exceeds max num " << PRIOR_BOX_MAX_NUM << ", got "
                  << prior_box_attr->GetAspectRatios();
    free(prior_box_param);
    return nullptr;
  }
  prior_box_param->aspect_ratios_size = prior_box_attr->GetAspectRatios().size();
  (void)memcpy(prior_box_param->aspect_ratios, prior_box_attr->GetAspectRatios().data(),
               prior_box_attr->GetAspectRatios().size() * sizeof(float));
  if (prior_box_attr->GetVariances().size() != PRIOR_BOX_VAR_NUM) {
    MS_LOG(ERROR) << "PriorBox variances size should be " << PRIOR_BOX_VAR_NUM << ", got "
                  << prior_box_attr->GetVariances().size();
    free(prior_box_param);
    return nullptr;
  }
  (void)memcpy(prior_box_param->variances, prior_box_attr->GetVariances().data(), PRIOR_BOX_VAR_NUM * sizeof(float));
  prior_box_param->flip = prior_box_attr->GetFlip();
  prior_box_param->clip = prior_box_attr->GetClip();
  prior_box_param->offset = prior_box_attr->GetOffset();
  prior_box_param->image_size_h = prior_box_attr->GetImageSizeH();
  prior_box_param->image_size_w = prior_box_attr->GetImageSizeW();
  prior_box_param->step_h = prior_box_attr->GetStepH();
  prior_box_param->step_w = prior_box_attr->GetStepW();
  return reinterpret_cast<OpParameter *>(prior_box_param);
}

OpParameter *PopulateLstmParameter(const mindspore::lite::PrimitiveC *primitive) {
  LstmParameter *lstm_param = reinterpret_cast<LstmParameter *>(malloc(sizeof(LstmParameter)));
  if (lstm_param == nullptr) {
    MS_LOG(ERROR) << "malloc LstmParameter failed.";
    return nullptr;
  }
  memset(lstm_param, 0, sizeof(LstmParameter));
  lstm_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Lstm *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  if (param == nullptr) {
    free(lstm_param);
    MS_LOG(ERROR) << "get Lstm param nullptr.";
    return nullptr;
  }
  lstm_param->bidirectional_ = param->GetBidirection();
  return reinterpret_cast<OpParameter *>(lstm_param);
}

OpParameter *PopulateEmbeddingLookupParameter(const mindspore::lite::PrimitiveC *primitive) {
  EmbeddingLookupParameter *embedding_lookup_parameter =
      reinterpret_cast<EmbeddingLookupParameter *>(malloc(sizeof(EmbeddingLookupParameter)));
  if (embedding_lookup_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc EmbeddingLookupParameter failed.";
    return nullptr;
  }
  memset(embedding_lookup_parameter, 0, sizeof(EmbeddingLookupParameter));
  embedding_lookup_parameter->op_parameter_.type_ = primitive->Type();
  auto param =
    reinterpret_cast<mindspore::lite::EmbeddingLookup *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  embedding_lookup_parameter->max_norm_ = param->GetMaxNorm();
  if (embedding_lookup_parameter->max_norm_ < 0) {
    MS_LOG(ERROR) << "Embedding lookup max norm should be positive number, got "
                  << embedding_lookup_parameter->max_norm_;
    free(embedding_lookup_parameter);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(embedding_lookup_parameter);
}

OpParameter *PopulateBiasAddParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *arithmetic_param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(arithmetic_param, 0, sizeof(ArithmeticParameter));
  arithmetic_param->op_parameter_.type_ = primitive->Type();

  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

OpParameter *PopulateEluParameter(const mindspore::lite::PrimitiveC *primitive) {
  EluParameter *elu_parameter = reinterpret_cast<EluParameter *>(malloc(sizeof(EluParameter)));
  if (elu_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc EluParameter failed.";
    return nullptr;
  }
  memset(elu_parameter, 0, sizeof(EluParameter));
  elu_parameter->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Elu *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  elu_parameter->alpha_ = param->GetAlpha();
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
  populate_parameter_funcs_[schema::PrimitiveType_ConstantOfShape] = PopulateConstantOfShapeParameter;
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
  populate_parameter_funcs_[schema::PrimitiveType_CaffePReLU] = PopulatePReLUParameter;
  populate_parameter_funcs_[schema::PrimitiveType_Prelu] = PopulateLeakyReluParameter;
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

PopulateParameterFunc PopulateParameterRegistry::GetParameterFunc(int type) {
  return populate_parameter_funcs_[schema::PrimitiveType(type)];
}

OpParameter *PopulateParameter(const mindspore::lite::PrimitiveC *primitive) {
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
