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
  auto *roi_pooling_param = new (std::nothrow) ROIPoolingParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "new PoolingParameter failed.";
    return nullptr;
  }
  roi_pooling_param->op_parameter_.type_ = primitive->Type();
  roi_pooling_param->pooledH_ = param->GetPooledW();
  roi_pooling_param->pooledW_ = param->GetPooledW();
  roi_pooling_param->scale_ = param->GetScale();
  return reinterpret_cast<OpParameter *>(roi_pooling_param);
}

OpParameter *PopulateBatchNorm(const mindspore::lite::PrimitiveC *primitive) {
  const auto param =
    reinterpret_cast<mindspore::lite::BatchNorm *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto *batch_norm_param = new (std::nothrow) BatchNormParameter();
  if (batch_norm_param == nullptr) {
    MS_LOG(ERROR) << "new BatchNormParameter failed.";
    return nullptr;
  }
  batch_norm_param->op_parameter_.type_ = primitive->Type();
  batch_norm_param->epsilon_ = param->GetEpsilon();
  batch_norm_param->fused_ = false;
  return reinterpret_cast<OpParameter *>(batch_norm_param);
}

OpParameter *PopulateFillParameter(const mindspore::lite::PrimitiveC *primitive) {
  const auto param = reinterpret_cast<mindspore::lite::Fill *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto *fill_param = new (std::nothrow) FillParameter();
  if (fill_param == nullptr) {
    MS_LOG(ERROR) << "new FillParameter failed.";
    return nullptr;
  }
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
  auto *expand_dims_param = new (std::nothrow) ExpandDimsParameter();
  if (expand_dims_param == nullptr) {
    MS_LOG(ERROR) << "new ExpandDimsParameter failed.";
    return nullptr;
  }
  expand_dims_param->op_parameter_.type_ = primitive->Type();
  expand_dims_param->dim_ = param->GetDim();
  return reinterpret_cast<OpParameter *>(expand_dims_param);
}

OpParameter *PopulatePReLUParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto param = dynamic_cast<const mindspore::lite::CaffePReLU *>(primitive);
  auto *prelu_param = new (std::nothrow) PReluParameter();
  if (prelu_param == nullptr) {
    MS_LOG(ERROR) << "new caffePReluParameter failed.";
    return nullptr;
  }
  prelu_param->op_parameter_.type_ = primitive->Type();
  prelu_param->channelShared = param->GetChannelShared();
  return reinterpret_cast<OpParameter *>(prelu_param);
}

OpParameter *PopulateLeakyReluParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto param = dynamic_cast<const mindspore::lite::Prelu *>(primitive);
  LeakyReluParameter *leaky_relu_param = new (std::nothrow) LeakyReluParameter();
  if (leaky_relu_param == nullptr) {
    MS_LOG(ERROR) << "new LeakyReluParameter failed.";
    return nullptr;
  }
  leaky_relu_param->op_parameter_.type_ = primitive->Type();
  auto temp = param->GetSlope();
  leaky_relu_param->slope_ = reinterpret_cast<float *>(malloc(temp.size() * sizeof(float)));
  if (leaky_relu_param->slope_ == nullptr) {
    MS_LOG(ERROR) << "malloc relu slope fail!";
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
  auto *pooling_param = new (std::nothrow) PoolingParameter();
  if (pooling_param == nullptr) {
    MS_LOG(ERROR) << "new PoolingParameter failed.";
    return nullptr;
  }
  pooling_param->op_parameter_.type_ = primitive->Type();
  pooling_param->global_ = pooling_primitive->GetGlobal();
  pooling_param->window_w_ = pooling_primitive->GetWindowW();
  pooling_param->window_h_ = pooling_primitive->GetWindowH();
  auto pooling_lite_primitive = (lite::Pooling *)primitive;
  MS_ASSERT(nullptr != pooling_lite_primitive);
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
  auto *matmul_param = new (std::nothrow) MatMulParameter();
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "new FullconnectionParameter failed.";
    return nullptr;
  }
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
  auto *matmul_param = new (std::nothrow) MatMulParameter();
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "new FullconnectionParameter failed.";
    return nullptr;
  }
  matmul_param->op_parameter_.type_ = primitive->Type();
  matmul_param->b_transpose_ = param->GetTransposeB();
  matmul_param->a_transpose_ = param->GetTransposeA();
  matmul_param->has_bias_ = false;
  matmul_param->act_type_ = ActType_No;
  return reinterpret_cast<OpParameter *>(matmul_param);
}

OpParameter *PopulateConvParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *conv_param = new (std::nothrow) ConvParameter();
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }
  conv_param->op_parameter_.type_ = primitive->Type();
  auto conv_primitive =
    reinterpret_cast<mindspore::lite::Conv2D *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  conv_param->kernel_h_ = conv_primitive->GetKernelH();
  conv_param->kernel_w_ = conv_primitive->GetKernelW();
  conv_param->group_ = conv_primitive->GetGroup();
  conv_param->stride_h_ = conv_primitive->GetStrideH();
  conv_param->stride_w_ = conv_primitive->GetStrideW();

  auto conv2d_lite_primitive = (lite::Conv2D *)primitive;
  MS_ASSERT(nullptr != conv2d_lite_primitive);
  conv_param->pad_u_ = conv2d_lite_primitive->PadUp();
  conv_param->pad_d_ = conv2d_lite_primitive->PadDown();
  conv_param->pad_l_ = conv2d_lite_primitive->PadLeft();
  conv_param->pad_r_ = conv2d_lite_primitive->PadRight();
  conv_param->dilation_h_ = conv_primitive->GetDilateH();
  conv_param->dilation_w_ = conv_primitive->GetDilateW();
  conv_param->input_channel_ = conv_primitive->GetChannelIn();
  conv_param->output_channel_ = conv_primitive->GetChannelOut();
  conv_param->group_ = conv_primitive->GetGroup();
  auto act_type = conv_primitive->GetActivationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      conv_param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      conv_param->act_type_ = ActType_Relu6;
      break;
    default:
      conv_param->act_type_ = ActType_No;
      break;
  }
  return reinterpret_cast<OpParameter *>(conv_param);
}

OpParameter *PopulateConvDwParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *conv_param = new (std::nothrow) ConvParameter();
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }
  conv_param->op_parameter_.type_ = primitive->Type();

  auto conv_primitive =
    reinterpret_cast<mindspore::lite::DepthwiseConv2D *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  conv_param->kernel_h_ = conv_primitive->GetKernelH();
  conv_param->kernel_w_ = conv_primitive->GetKernelW();
  conv_param->stride_h_ = conv_primitive->GetStrideH();
  conv_param->stride_w_ = conv_primitive->GetStrideW();

  auto convdw_lite_primitive = (lite::DepthwiseConv2D *)primitive;
  MS_ASSERT(nullptr != convdw_lite_primitive);
  conv_param->pad_u_ = convdw_lite_primitive->PadUp();
  conv_param->pad_d_ = convdw_lite_primitive->PadDown();
  conv_param->pad_l_ = convdw_lite_primitive->PadLeft();
  conv_param->pad_r_ = convdw_lite_primitive->PadRight();
  conv_param->dilation_h_ = conv_primitive->GetDilateH();
  conv_param->dilation_w_ = conv_primitive->GetDilateW();
  auto act_type = conv_primitive->GetActivationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      conv_param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      conv_param->act_type_ = ActType_Relu6;
      break;
    default:
      conv_param->act_type_ = ActType_No;
      break;
  }
  return reinterpret_cast<OpParameter *>(conv_param);
}

OpParameter *PopulateDeconvDwParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *conv_param = new ConvParameter();
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }
  conv_param->op_parameter_.type_ = primitive->Type();
  auto conv_primitive =
    reinterpret_cast<mindspore::lite::DeDepthwiseConv2D *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  conv_param->kernel_h_ = conv_primitive->GetKernelH();
  conv_param->kernel_w_ = conv_primitive->GetKernelW();
  conv_param->stride_h_ = conv_primitive->GetStrideH();
  conv_param->stride_w_ = conv_primitive->GetStrideW();

  auto deconvdw_lite_primitive = (mindspore::lite::DeDepthwiseConv2D *)primitive;
  MS_ASSERT(nullptr != deconvdw_lite_primitive);
  conv_param->pad_u_ = deconvdw_lite_primitive->PadUp();
  conv_param->pad_d_ = deconvdw_lite_primitive->PadDown();
  conv_param->pad_l_ = deconvdw_lite_primitive->PadLeft();
  conv_param->pad_r_ = deconvdw_lite_primitive->PadRight();
  conv_param->dilation_h_ = conv_primitive->GetDilateH();
  conv_param->dilation_w_ = conv_primitive->GetDilateW();
  auto act_type = conv_primitive->GetActivationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      conv_param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      conv_param->act_type_ = ActType_Relu6;
      break;
    default:
      conv_param->act_type_ = ActType_No;
      break;
  }
  return reinterpret_cast<OpParameter *>(conv_param);
}

OpParameter *PopulateDeconvParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *conv_param = new ConvParameter();
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }
  conv_param->op_parameter_.type_ = primitive->Type();
  auto conv_primitive =
    reinterpret_cast<mindspore::lite::DeConv2D *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  conv_param->kernel_h_ = conv_primitive->GetKernelH();
  conv_param->kernel_w_ = conv_primitive->GetKernelW();
  conv_param->stride_h_ = conv_primitive->GetStrideH();
  conv_param->stride_w_ = conv_primitive->GetStrideW();

  auto deconv_lite_primitive = (lite::DeConv2D *)primitive;
  MS_ASSERT(nullptr != deconvdw_lite_primitive);
  conv_param->pad_u_ = deconv_lite_primitive->PadUp();
  conv_param->pad_d_ = deconv_lite_primitive->PadDown();
  conv_param->pad_l_ = deconv_lite_primitive->PadLeft();
  conv_param->pad_r_ = deconv_lite_primitive->PadRight();
  conv_param->dilation_h_ = conv_primitive->GetDilateH();
  conv_param->dilation_w_ = conv_primitive->GetDilateW();
  auto act_type = conv_primitive->GetActivationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      conv_param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      conv_param->act_type_ = ActType_Relu6;
      break;
    default:
      conv_param->act_type_ = ActType_No;
      break;
  }
  return reinterpret_cast<OpParameter *>(conv_param);
}

OpParameter *PopulateSoftmaxParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto softmax_primitive =
    reinterpret_cast<mindspore::lite::SoftMax *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto *softmax_param = new (std::nothrow) SoftmaxParameter();
  if (softmax_param == nullptr) {
    MS_LOG(ERROR) << "new SoftmaxParameter failed.";
    return nullptr;
  }
  softmax_param->op_parameter_.type_ = primitive->Type();
  softmax_param->axis_ = softmax_primitive->GetAxis();
  return reinterpret_cast<OpParameter *>(softmax_param);
}

OpParameter *PopulateReduceParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *reduce_param = new (std::nothrow) ReduceParameter();
  if (reduce_param == nullptr) {
    MS_LOG(ERROR) << "new ReduceParameter failed.";
    return nullptr;
  }
  reduce_param->op_parameter_.type_ = primitive->Type();
  auto reduce = reinterpret_cast<mindspore::lite::Reduce *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  reduce_param->keep_dims_ = reduce->GetKeepDims();
  auto axisVector = reduce->GetAxes();
  if (axisVector.size() > REDUCE_MAX_AXES_NUM) {
    MS_LOG(ERROR) << "Reduce axes size " << axisVector.size() << " exceed limit " << REDUCE_MAX_AXES_NUM;
    delete (reduce_param);
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
  auto *mean_param = new (std::nothrow) ReduceParameter();
  if (mean_param == nullptr) {
    MS_LOG(ERROR) << "new ReduceParameter failed.";
    return nullptr;
  }
  mean_param->op_parameter_.type_ = primitive->Type();
  auto mean = reinterpret_cast<mindspore::lite::Mean *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  mean_param->keep_dims_ = mean->GetKeepDims();
  auto axisVector = mean->GetAxis();
  if (axisVector.size() > REDUCE_MAX_AXES_NUM) {
    MS_LOG(ERROR) << "Reduce axes size " << axisVector.size() << " exceed limit " << REDUCE_MAX_AXES_NUM;
    delete (mean_param);
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
  auto *pad_param = new (std::nothrow) PadParameter();
  if (pad_param == nullptr) {
    MS_LOG(ERROR) << "new PadParameter failed.";
    return nullptr;
  }
  pad_param->op_parameter_.type_ = primitive->Type();
  auto pad_node = reinterpret_cast<mindspore::lite::Pad *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  pad_param->pad_mode_ = pad_node->GetPaddingMode();
  if (pad_param->pad_mode_ == schema::PaddingMode_CONSTANT) {
    pad_param->constant_value_ = pad_node->GetConstantValue();
  } else {
    MS_LOG(ERROR) << "Invalid padding mode: " << pad_param->pad_mode_;
    delete (pad_param);
    return nullptr;
  }

  auto size = pad_node->GetPaddings().size();
  if (size > MAX_PAD_SIZE) {
    MS_LOG(ERROR) << "Invalid padding size: " << size;
    delete (pad_param);
    return nullptr;
  }

  for (size_t i = 0; i < size; i++) {
    pad_param->paddings_[MAX_PAD_SIZE - size + i] = pad_node->GetPaddings()[i];
  }
  return reinterpret_cast<OpParameter *>(pad_param);
}

OpParameter *PopulateActivationParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *act_param = new (std::nothrow) ActivationParameter();
  if (act_param == nullptr) {
    MS_LOG(ERROR) << "new ActivationParameter failed.";
    return nullptr;
  }
  auto activation =
    reinterpret_cast<mindspore::lite::Activation *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  act_param->type_ = static_cast<int>(activation->GetType());
  act_param->alpha_ = activation->GetAlpha();
  return reinterpret_cast<OpParameter *>(act_param);
}

OpParameter *PopulateFusedBatchNorm(const mindspore::lite::PrimitiveC *primitive) {
  auto *batch_norm_param = new (std::nothrow) BatchNormParameter();
  if (batch_norm_param == nullptr) {
    MS_LOG(ERROR) << "new FusedBatchNormParameter failed.";
    return nullptr;
  }
  batch_norm_param->op_parameter_.type_ = primitive->Type();
  auto param =
    reinterpret_cast<mindspore::lite::FusedBatchNorm *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  batch_norm_param->epsilon_ = param->GetEpsilon();
  batch_norm_param->fused_ = true;
  return reinterpret_cast<OpParameter *>(batch_norm_param);
}

OpParameter *PopulateArithmetic(const mindspore::lite::PrimitiveC *primitive) {
  auto *arithmetic_param = new (std::nothrow) ArithmeticParameter();
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticParameter failed.";
    return nullptr;
  }
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
  auto *arithmetic_param = new (std::nothrow) ArithmeticParameter();
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticParameter failed.";
    return nullptr;
  }
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
      delete arithmetic_param;
      return nullptr;
  }
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

OpParameter *PopulateArithmeticSelf(const mindspore::lite::PrimitiveC *primitive) {
  auto *arithmetic_self_param = new (std::nothrow) ArithmeticSelfParameter();
  if (arithmetic_self_param == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticParameter failed.";
    return nullptr;
  }
  arithmetic_self_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(arithmetic_self_param);
}

OpParameter *PopulatePowerParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *power_param = new (std::nothrow) PowerParameter();
  if (power_param == nullptr) {
    MS_LOG(ERROR) << "new PowerParameter failed.";
    return nullptr;
  }
  power_param->op_parameter_.type_ = primitive->Type();
  auto power = reinterpret_cast<mindspore::lite::Power *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  power_param->power_ = power->GetPower();
  power_param->scale_ = power->GetScale();
  power_param->shift_ = power->GetShift();
  return reinterpret_cast<OpParameter *>(power_param);
}

OpParameter *PopulateArgMaxParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *arg_param = new (std::nothrow) ArgMinMaxParameter();
  if (arg_param == nullptr) {
    MS_LOG(ERROR) << "new ArgMinMaxParameter failed.";
    return nullptr;
  }
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
  auto *arg_param = new (std::nothrow) ArgMinMaxParameter();
  if (arg_param == nullptr) {
    MS_LOG(ERROR) << "new ArgMinMaxParameter failed.";
    return nullptr;
  }
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
  auto *cast_param = new (std::nothrow) CastParameter();
  if (cast_param == nullptr) {
    MS_LOG(ERROR) << "new CastParameter failed.";
    return nullptr;
  }
  cast_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Cast *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  cast_param->src_type_ = param->GetSrcT();
  cast_param->dst_type_ = param->GetDstT();
  return reinterpret_cast<OpParameter *>(cast_param);
}

OpParameter *PopulateLocalResponseNormParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto local_response_norm_attr = reinterpret_cast<mindspore::lite::LocalResponseNormalization *>(
    const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto *lrn_param = new (std::nothrow) LocalResponseNormParameter();
  if (lrn_param == nullptr) {
    MS_LOG(ERROR) << "new LocalResponseNormParameter failed.";
    return nullptr;
  }
  lrn_param->op_parameter_.type_ = primitive->Type();
  lrn_param->depth_radius_ = local_response_norm_attr->GetDepthRadius();
  lrn_param->bias_ = local_response_norm_attr->GetBias();
  lrn_param->alpha_ = local_response_norm_attr->GetAlpha();
  lrn_param->beta_ = local_response_norm_attr->GetBeta();
  return reinterpret_cast<OpParameter *>(lrn_param);
}

OpParameter *PopulateRangeParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto range_attr = reinterpret_cast<mindspore::lite::Range *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto *range_param = new (std::nothrow) RangeParameter();
  if (range_param == nullptr) {
    MS_LOG(ERROR) << "new RangeParameter failed.";
    return nullptr;
  }
  range_param->op_parameter_.type_ = primitive->Type();
  range_param->start_ = range_attr->GetStart();
  range_param->limit_ = range_attr->GetLimit();
  range_param->delta_ = range_attr->GetDelta();
  range_param->dType_ = range_attr->GetDType();
  return reinterpret_cast<OpParameter *>(range_param);
}

OpParameter *PopulateConcatParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *concat_param = new (std::nothrow) ConcatParameter();
  if (concat_param == nullptr) {
    MS_LOG(ERROR) << "new ConcatParameter failed.";
    return nullptr;
  }
  concat_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Concat *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  concat_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(concat_param);
}

OpParameter *PopulateTileParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *tile_param = new (std::nothrow) TileParameter();
  if (tile_param == nullptr) {
    MS_LOG(ERROR) << "new TileParameter failed.";
    return nullptr;
  }
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
  auto *topk_param = new (std::nothrow) TopkParameter();
  if (topk_param == nullptr) {
    MS_LOG(ERROR) << "new TopkParameter failed.";
    return nullptr;
  }
  topk_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::TopK *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  topk_param->k_ = param->GetK();
  topk_param->sorted_ = param->GetSorted();
  return reinterpret_cast<OpParameter *>(topk_param);
}

OpParameter *PopulateNhwc2NchwParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *parameter = new (std::nothrow) OpParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new Nhwc2NchwParameter failed.";
    return nullptr;
  }
  parameter->type_ = primitive->Type();
  return parameter;
}

OpParameter *PopulateNchw2NhwcParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *parameter = new (std::nothrow) OpParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new Nchw2NhwcParameter failed.";
    return nullptr;
  }
  parameter->type_ = primitive->Type();
  return parameter;
}

OpParameter *PopulateTransposeParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *transpose_param = new (std::nothrow) TransposeParameter();
  if (transpose_param == nullptr) {
    MS_LOG(ERROR) << "new TransposeParameter failed.";
    return nullptr;
  }
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
  auto *split_param = new (std::nothrow) SplitParameter();
  if (split_param == nullptr) {
    MS_LOG(ERROR) << "new SplitParameter failed.";
    return nullptr;
  }
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
  auto *squeeze_param = new (std::nothrow) SqueezeParameter();
  if (squeeze_param == nullptr) {
    MS_LOG(ERROR) << "new SqueezeParameter failed.";
    return nullptr;
  }
  squeeze_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(squeeze_param);
}

OpParameter *PopulateScaleParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "input primitive is nullptr";
    return nullptr;
  }
  auto *scale_param = new (std::nothrow) ScaleParameter();
  if (scale_param == nullptr) {
    MS_LOG(ERROR) << "new ScaleParameter failed.";
    return nullptr;
  }
  scale_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Scale *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  scale_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(scale_param);
}

OpParameter *PopulateGatherParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto gather_attr = reinterpret_cast<mindspore::lite::Gather *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto *gather_param = new (std::nothrow) GatherParameter();
  if (gather_param == nullptr) {
    MS_LOG(ERROR) << "new GatherParameter failed.";
    return nullptr;
  }
  gather_param->op_parameter_.type_ = primitive->Type();
  gather_param->axis_ = gather_attr->GetAxis();
  gather_param->batchDims_ = gather_attr->GetBatchDims();
  return reinterpret_cast<OpParameter *>(gather_param);
}

OpParameter *PopulateGatherNdParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *gather_nd_param = new (std::nothrow) GatherNdParameter();
  if (gather_nd_param == nullptr) {
    MS_LOG(ERROR) << "new GatherNDParameter failed.";
    return nullptr;
  }
  gather_nd_param->op_parameter_.type_ = primitive->Type();
  auto gatherNd_attr =
    reinterpret_cast<mindspore::lite::GatherNd *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  gather_nd_param->batchDims_ = gatherNd_attr->GetBatchDims();
  return reinterpret_cast<OpParameter *>(gather_nd_param);
}

OpParameter *PopulateScatterNDParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *scatter_nd_param = new (std::nothrow) ScatterNDParameter();
  if (scatter_nd_param == nullptr) {
    MS_LOG(ERROR) << "new ScatterNDParameter failed.";
    return nullptr;
  }
  scatter_nd_param->op_parameter_.type_ = primitive->Type();
  MS_ASSERT(paramter != nullptr);
  return reinterpret_cast<OpParameter *>(scatter_nd_param);
}

OpParameter *PopulateSliceParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *slice_param = new (std::nothrow) SliceParameter();
  if (slice_param == nullptr) {
    MS_LOG(ERROR) << "new SliceParameter failed.";
    return nullptr;
  }
  auto param = reinterpret_cast<mindspore::lite::Slice *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  slice_param->op_parameter_.type_ = primitive->Type();
  auto param_begin = param->GetBegin();
  auto param_size = param->GetSize();
  if (param_begin.size() != param_size.size()) {
    delete slice_param;
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
  auto *broadcast_param = new (std::nothrow) BroadcastToParameter();
  if (broadcast_param == nullptr) {
    MS_LOG(ERROR) << "new BroadcastToParameter failed.";
    return nullptr;
  }
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
  auto *reshape_param = new (std::nothrow) ReshapeParameter();
  if (reshape_param == nullptr) {
    MS_LOG(ERROR) << "new ReshapeParameter failed.";
    return nullptr;
  }
  reshape_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(reshape_param);
}

OpParameter *PopulateShapeParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *shape_param = new (std::nothrow) ShapeParameter();
  if (shape_param == nullptr) {
    MS_LOG(ERROR) << "new ShapeParameter failed.";
    return nullptr;
  }
  shape_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(shape_param);
}

OpParameter *PopulateConstantOfShapeParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto attr =
    reinterpret_cast<mindspore::lite::ConstantOfShape *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  ConstantOfShapeParameter *param = new (std::nothrow) ConstantOfShapeParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ConstantOfShapeParameter failed.";
    return nullptr;
  }
  param->op_parameter_.type_ = primitive->Type();
  param->value_ = attr->GetValue();
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateReverseParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto reverse_attr =
    reinterpret_cast<mindspore::lite::Reverse *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  ReverseParameter *reverse_param = new (std::nothrow) ReverseParameter();
  if (reverse_param == nullptr) {
    MS_LOG(ERROR) << "new ReverseParameter failed.";
    return nullptr;
  }
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
  auto *unsqueeze_param = new (std::nothrow) UnsqueezeParameter();
  if (unsqueeze_param == nullptr) {
    MS_LOG(ERROR) << "new ReverseParameter failed.";
    return nullptr;
  }
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
  auto *stack_param = new (std::nothrow) StackParameter();
  if (stack_param == nullptr) {
    MS_LOG(ERROR) << "new StackParameter failed.";
    return nullptr;
  }
  auto param = reinterpret_cast<mindspore::lite::Stack *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  stack_param->op_parameter_.type_ = primitive->Type();
  stack_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(stack_param);
}

OpParameter *PopulateUnstackParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *unstack_param = new (std::nothrow) UnstackParameter();
  if (unstack_param == nullptr) {
    MS_LOG(ERROR) << "new UnstackParameter failed.";
    return nullptr;
  }
  auto param = reinterpret_cast<mindspore::lite::Unstack *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  unstack_param->op_parameter_.type_ = primitive->Type();
  unstack_param->num_ = param->GetNum();
  unstack_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(unstack_param);
}

OpParameter *PopulateReverseSequenceParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *reverse_sequence_param = new (std::nothrow) ReverseSequenceParameter();
  if (reverse_sequence_param == nullptr) {
    MS_LOG(ERROR) << "new ReverseSequenceParameter failed.";
    return nullptr;
  }
  auto param =
    reinterpret_cast<mindspore::lite::ReverseSequence *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  reverse_sequence_param->op_parameter_.type_ = primitive->Type();
  reverse_sequence_param->seq_axis_ = param->GetSeqAxis();
  reverse_sequence_param->batch_axis_ = param->GetBatchAxis();
  return reinterpret_cast<OpParameter *>(reverse_sequence_param);
}

OpParameter *PopulateUniqueParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *unique_param = new (std::nothrow) UniqueParameter();
  if (unique_param == nullptr) {
    MS_LOG(ERROR) << "new PopulateUniqueParam failed.";
    return nullptr;
  }
  unique_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(unique_param);
}

OpParameter *PopulateDepthToSpaceParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *depth_space_param = new (std::nothrow) DepthToSpaceParameter();
  if (depth_space_param == nullptr) {
    MS_LOG(ERROR) << "new DepthToSpaceParameter failed.";
    return nullptr;
  }
  auto param = reinterpret_cast<mindspore::lite::DepthToSpace *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  depth_space_param->op_parameter_.type_ = primitive->Type();
  depth_space_param->block_size_ = param->GetBlockSize();
  return reinterpret_cast<OpParameter *>(depth_space_param);
}

OpParameter *PopulateSpaceToDepthParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *space_depth_param = new (std::nothrow) SpaceToDepthParameter();
  if (space_depth_param == nullptr) {
    MS_LOG(ERROR) << "new SpaceToDepthspace_depth_param failed.";
    return nullptr;
  }
  space_depth_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::SpaceToDepth *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  space_depth_param->op_parameter_.type_ = primitive->Type();
  space_depth_param->block_size_ = param->GetBlockSize();
  if (param->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "Currently only NHWC format is supported.";
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(space_depth_param);
}

OpParameter *PopulateSpaceToBatchParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *space_batch_param = new (std::nothrow) SpaceToBatchParameter();
  if (space_batch_param == nullptr) {
    MS_LOG(ERROR) << "new SpaceToBatchParameter failed.";
    return nullptr;
  }
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
  auto *resize_param = new (std::nothrow) ResizeParameter();
  if (resize_param == nullptr) {
    MS_LOG(ERROR) << "new ResizeParameter failed.";
    return nullptr;
  }
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
  auto *batch_space_param = new (std::nothrow) BatchToSpaceParameter();
  if (batch_space_param == nullptr) {
    MS_LOG(ERROR) << "New BatchToSpaceParameter fail!";
    return nullptr;
  }
  batch_space_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::BatchToSpace *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto block_shape = param->GetBlockShape();
  if (block_shape.size() != BATCH_TO_SPACE_BLOCK_SHAPE_SIZE) {
    MS_LOG(ERROR) << "batch_to_space blockShape size should be " << BATCH_TO_SPACE_BLOCK_SHAPE_SIZE;
    return nullptr;
  }

  auto crops = param->GetCrops();
  if (crops.size() != BATCH_TO_SPACE_CROPS_SIZE) {
    MS_LOG(ERROR) << "batch_to_space crops size should be " << BATCH_TO_SPACE_CROPS_SIZE;
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
  auto *crop_param = new (std::nothrow) CropParameter();
  if (crop_param == nullptr) {
    MS_LOG(ERROR) << "new CropParameter fail!";
    return nullptr;
  }
  crop_param->op_parameter_.type_ = primitive->Type();
  crop_param->axis_ = param->GetAxis();
  crop_param->offset_size_ = param_offset.size();
  for (size_t i = 0; i < param_offset.size(); ++i) {
    crop_param->offset_[i] = param_offset[i];
  }
  return reinterpret_cast<OpParameter *>(crop_param);
}

OpParameter *PopulateOneHotParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *one_hot_param = new (std::nothrow) OneHotParameter();
  if (one_hot_param == nullptr) {
    MS_LOG(ERROR) << "new OneHotParameter fail!";
    return nullptr;
  }
  one_hot_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::OneHot *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  if (param == nullptr) {
    delete (one_hot_param);
    MS_LOG(ERROR) << "get OneHot param nullptr.";
    return nullptr;
  }
  one_hot_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(one_hot_param);
}

OpParameter *PopulateFlattenParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *flatten_param = new (std::nothrow) FlattenParameter();
  if (flatten_param == nullptr) {
    MS_LOG(ERROR) << "new FlattenParameter fail!";
    return nullptr;
  }
  flatten_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(flatten_param);
}

OpParameter *PopulateQuantDTypeCastParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *parameter = new (std::nothrow) QuantDTypeCastParameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "new QuantDTypeCastParameter fail!";
    return nullptr;
  }
  parameter->op_parameter_.type_ = primitive->Type();
  auto quant_dtype_cast_param =
    reinterpret_cast<mindspore::lite::QuantDTypeCast *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  parameter->srcT = quant_dtype_cast_param->GetSrcT();
  parameter->dstT = quant_dtype_cast_param->GetDstT();
  return reinterpret_cast<OpParameter *>(parameter);
}

OpParameter *PopulateStridedSliceParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *strided_slice_param = new (std::nothrow) StridedSliceParameter();
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

OpParameter *PopulateAddNParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto addn_param = new (std::nothrow) OpParameter();
  if (addn_param == nullptr) {
    MS_LOG(ERROR) << "new OpParameter fail!";
    return nullptr;
  }
  addn_param->type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(addn_param);
}

OpParameter *PopulatePriorBoxParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *prior_box_param = new (std::nothrow) PriorBoxParameter();
  if (prior_box_param == nullptr) {
    MS_LOG(ERROR) << "new PriorBoxParameter failed.";
    return nullptr;
  }
  prior_box_param->op_parameter_.type_ = primitive->Type();
  auto prior_box_attr =
    reinterpret_cast<mindspore::lite::PriorBox *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));

  if (prior_box_attr->GetMinSizes().size() > PRIOR_BOX_MAX_NUM) {
    MS_LOG(ERROR) << "PriorBox min_sizes size exceeds max num " << PRIOR_BOX_MAX_NUM << ", got "
                  << prior_box_attr->GetMinSizes();
    delete (prior_box_param);
    return nullptr;
  }
  prior_box_param->min_sizes_size = prior_box_attr->GetMinSizes().size();
  if (prior_box_attr->GetMaxSizes().size() > PRIOR_BOX_MAX_NUM) {
    MS_LOG(ERROR) << "PriorBox max_sizes size exceeds max num " << PRIOR_BOX_MAX_NUM << ", got "
                  << prior_box_attr->GetMaxSizes();
    delete (prior_box_param);
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
    delete (prior_box_param);
    return nullptr;
  }
  prior_box_param->aspect_ratios_size = prior_box_attr->GetAspectRatios().size();
  (void)memcpy(prior_box_param->aspect_ratios, prior_box_attr->GetAspectRatios().data(),
               prior_box_attr->GetAspectRatios().size() * sizeof(float));
  if (prior_box_attr->GetVariances().size() != PRIOR_BOX_VAR_NUM) {
    MS_LOG(ERROR) << "PriorBox variances size should be " << PRIOR_BOX_VAR_NUM << ", got "
                  << prior_box_attr->GetVariances().size();
    delete (prior_box_param);
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
  auto *lstm_param = new (std::nothrow) LstmParameter();
  if (lstm_param == nullptr) {
    MS_LOG(ERROR) << "new LstmParameter fail!";
    return nullptr;
  }
  lstm_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Lstm *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  if (param == nullptr) {
    delete (lstm_param);
    MS_LOG(ERROR) << "get Lstm param nullptr.";
    return nullptr;
  }
  lstm_param->bidirectional_ = param->GetBidirection();
  return reinterpret_cast<OpParameter *>(lstm_param);
}

OpParameter *PopulateEmbeddingLookupParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *embedding_lookup_parameter = new (std::nothrow) EmbeddingLookupParameter();
  if (embedding_lookup_parameter == nullptr) {
    MS_LOG(ERROR) << "new EmbeddingLookupParameter failed";
    return nullptr;
  }
  embedding_lookup_parameter->op_parameter_.type_ = primitive->Type();
  auto param =
    reinterpret_cast<mindspore::lite::EmbeddingLookup *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  embedding_lookup_parameter->max_norm_ = param->GetMaxNorm();
  if (embedding_lookup_parameter->max_norm_ < 0) {
    MS_LOG(ERROR) << "Embedding lookup max norm should be positive number, got "
                  << embedding_lookup_parameter->max_norm_;
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(embedding_lookup_parameter);
}

OpParameter *PopulateBiasAddParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *arithmetic_param = new (std::nothrow) ArithmeticParameter();
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "new Bias Add Parameter failed";
    return nullptr;
  }
  arithmetic_param->op_parameter_.type_ = primitive->Type();

  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

OpParameter *PopulateEluParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *elu_parameter = new (std::nothrow) EluParameter();
  if (elu_parameter == nullptr) {
    MS_LOG(ERROR) << "new EluParameter failed";
    return nullptr;
  }
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
