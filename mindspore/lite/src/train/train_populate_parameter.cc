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

#include "src/train/train_populate_parameter.h"
#include <algorithm>
#include "src/ops/populate/populate_register.h"
#include "src/ops/pooling_grad.h"
#include "nnacl/pooling_parameter.h"
#include "src/ops/softmax_cross_entropy.h"
#include "src/ops/sparse_softmax_cross_entropy.h"
#include "nnacl/fp32_grad/softmax_grad.h"
#include "src/ops/activation_grad.h"
#include "nnacl/fp32/activation_fp32.h"
#include "src/ops/conv2d_grad_filter.h"
#include "src/ops/conv2d_grad_input.h"
#include "src/ops/group_conv2d_grad_input.h"
#include "nnacl/conv_parameter.h"
#include "src/ops/power_grad.h"
#include "nnacl/power_parameter.h"
#include "src/ops/bias_grad.h"
#include "nnacl/arithmetic.h"
#include "nnacl/fp32_grad/optimizer.h"
#include "src/ops/apply_momentum.h"
#include "src/ops/sgd.h"
#include "src/ops/bn_grad.h"
#include "nnacl/fp32_grad/batch_norm.h"
#include "src/ops/adam.h"
#include "nnacl/fp32_grad/dropout_parameter.h"
#include "src/ops/dropout.h"
#include "src/ops/dropout_grad.h"
#include "src/ops/arithmetic.h"
#include "src/ops/oneslike.h"
#include "src/ops/binary_cross_entropy.h"
#include "src/ops/binary_cross_entropy_grad.h"
#include "src/ops/smooth_l1_loss.h"
#include "src/ops/smooth_l1_loss_grad.h"
#include "nnacl/fp32_grad/smooth_l1_loss.h"
#include "src/ops/arithmetic_grad.h"
namespace mindspore::kernel {

OpParameter *DefaultPopulateParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }

  OpParameter *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc Param for primitive failed.";
    return nullptr;
  }

  param->type_ = primitive->Type();
  return param;
}

OpParameter *PopulateSmoothL1LossParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  SmoothL1LossParameter *p = reinterpret_cast<SmoothL1LossParameter *>(malloc(sizeof(SmoothL1LossParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "malloc SmoothL1LossParameter failed.";
    return nullptr;
  }
  p->op_parameter_.type_ = primitive->Type();

  auto smooth_l1_primitive =
    reinterpret_cast<mindspore::lite::SmoothL1Loss *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));

  p->beta_ = smooth_l1_primitive->GetBeta();
  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateSmoothL1LossGradParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  SmoothL1LossParameter *p = reinterpret_cast<SmoothL1LossParameter *>(malloc(sizeof(SmoothL1LossParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "malloc SmoothL1LossParameter failed.";
    return nullptr;
  }
  p->op_parameter_.type_ = primitive->Type();

  auto smooth_l1_primitive =
    reinterpret_cast<mindspore::lite::SmoothL1LossGrad *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));

  p->beta_ = smooth_l1_primitive->GetBeta();
  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateApplyMomentumParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  ApplyMomentumParameter *p = reinterpret_cast<ApplyMomentumParameter *>(malloc(sizeof(ApplyMomentumParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "malloc ApplyMomentumParameter failed.";
    return nullptr;
  }
  p->op_parameter_.type_ = primitive->Type();

  auto apply_momentum_primitive =
    reinterpret_cast<mindspore::lite::ApplyMomentum *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));

  p->grad_scale_ = apply_momentum_primitive->GetGradientScale();
  p->use_nesterov_ = apply_momentum_primitive->GetUseNesterov();

  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateBCEParameter(const mindspore::lite::PrimitiveC *primitive) {
  int32_t *reduction = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  if (reduction == nullptr) {
    MS_LOG(ERROR) << "malloc reduction failed.";
    return nullptr;
  }
  auto param =
    reinterpret_cast<mindspore::lite::BinaryCrossEntropy *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  *reduction = param->GetReduction();
  return reinterpret_cast<OpParameter *>(reduction);
}

OpParameter *PopulateBCEGradParameter(const mindspore::lite::PrimitiveC *primitive) {
  int32_t *reduction = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  if (reduction == nullptr) {
    MS_LOG(ERROR) << "malloc reduction failed.";
    return nullptr;
  }
  auto param =
    reinterpret_cast<mindspore::lite::BinaryCrossEntropyGrad *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  *reduction = param->GetReduction();
  return reinterpret_cast<OpParameter *>(reduction);
}

OpParameter *PopulateAdamParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  AdamParameter *p = reinterpret_cast<AdamParameter *>(malloc(sizeof(AdamParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "new AdamParameter failed.";
    return nullptr;
  }
  p->op_parameter_.type_ = primitive->Type();

  auto apply_momentum_primitive =
    reinterpret_cast<mindspore::lite::Adam *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  p->use_nesterov_ = apply_momentum_primitive->GetUseNesterov();
  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateSgdParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  SgdParameter *p = reinterpret_cast<SgdParameter *>(malloc(sizeof(SgdParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "malloc SgdParameter failed.";
    return nullptr;
  }
  p->op_parameter_.type_ = primitive->Type();

  auto sgd_primitive = reinterpret_cast<mindspore::lite::Sgd *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));

  p->weight_decay_ = sgd_primitive->GetWeightDecay();
  p->dampening_ = sgd_primitive->GetDampening();
  p->use_nesterov_ = sgd_primitive->GetUseNesterov();

  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateSparseSoftmaxCrossEntropyParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  SoftmaxCrossEntropyParameter *sce_param =
    reinterpret_cast<SoftmaxCrossEntropyParameter *>(malloc(sizeof(SoftmaxCrossEntropyParameter)));
  if (sce_param == nullptr) {
    MS_LOG(ERROR) << "malloc SoftmaxCrossEntropyParameter failed.";
    return nullptr;
  }
  auto sce_primitive = reinterpret_cast<mindspore::lite::SparseSoftmaxCrossEntropy *>(
    const_cast<mindspore::lite::PrimitiveC *>(primitive));

  sce_param->is_grad = sce_primitive->GetIsGrad();

  sce_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(sce_param);
}

OpParameter *PopulateSoftmaxCrossEntropyParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  SoftmaxCrossEntropyParameter *sce_param =
    reinterpret_cast<SoftmaxCrossEntropyParameter *>(malloc(sizeof(SoftmaxCrossEntropyParameter)));
  if (sce_param == nullptr) {
    MS_LOG(ERROR) << "malloc SoftmaxCrossEntropyParameter failed.";
    return nullptr;
  }
  sce_param->is_grad = 0;
  sce_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(sce_param);
}

OpParameter *PopulatePoolingGradParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  PoolingParameter *pooling_param = reinterpret_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  if (pooling_param == nullptr) {
    MS_LOG(ERROR) << "malloc PoolingParameter failed.";
    return nullptr;
  }
  pooling_param->op_parameter_.type_ = primitive->Type();
  auto pooling_primitive =
    reinterpret_cast<mindspore::lite::PoolingGrad *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));

  pooling_param->global_ = pooling_primitive->GetGlobal();
  pooling_param->window_w_ = pooling_primitive->GetWindowW();
  pooling_param->window_h_ = pooling_primitive->GetWindowH();

  pooling_param->pad_u_ = pooling_primitive->GetPadUp();
  pooling_param->pad_d_ = pooling_primitive->GetPadDown();
  pooling_param->pad_l_ = pooling_primitive->GetPadLeft();
  pooling_param->pad_r_ = pooling_primitive->GetPadRight();
  pooling_param->stride_w_ = pooling_primitive->GetStrideW();
  pooling_param->stride_h_ = pooling_primitive->GetStrideH();

  pooling_param->pool_mode_ = PoolMode_No;
  pooling_param->round_mode_ = RoundMode_No;

  switch (pooling_primitive->GetPoolingMode()) {
    case schema::PoolMode_MAX_POOLING:
      pooling_param->pool_mode_ = PoolMode_MaxPool;
      break;
    case schema::PoolMode_MEAN_POOLING:
      pooling_param->pool_mode_ = PoolMode_AvgPool;
      break;
    default:
      break;
  }

  switch (pooling_primitive->GetRoundMode()) {
    case schema::RoundMode_FLOOR:
      pooling_param->round_mode_ = RoundMode_Floor;
      break;
    case schema::RoundMode_CEIL:
      pooling_param->round_mode_ = RoundMode_Ceil;
      break;
    default:
      break;
  }
  return reinterpret_cast<OpParameter *>(pooling_param);
}

OpParameter *PopulateActivationGradParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }

  ActivationParameter *act_param = reinterpret_cast<ActivationParameter *>(malloc(sizeof(ActivationParameter)));
  if (act_param == nullptr) {
    MS_LOG(ERROR) << "malloc ActivationParameter failed.";
    return nullptr;
  }
  act_param->op_parameter_.type_ = primitive->Type();
  auto activation =
    reinterpret_cast<mindspore::lite::ActivationGrad *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  act_param->type_ = static_cast<int>(activation->GetType());
  act_param->alpha_ = activation->GetAlpha();
  return reinterpret_cast<OpParameter *>(act_param);
}

OpParameter *PopulateConvolutionGradFilterParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }

  ConvParameter *param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc Param for conv grad filter failed.";
    return nullptr;
  }
  param->op_parameter_.type_ = primitive->Type();

  auto convg_primitive =
    reinterpret_cast<mindspore::lite::Conv2DGradFilter *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  param->kernel_h_ = convg_primitive->GetKernelH();
  param->kernel_w_ = convg_primitive->GetKernelW();
  param->stride_h_ = convg_primitive->GetStrideH();
  param->stride_w_ = convg_primitive->GetStrideW();
  param->dilation_h_ = convg_primitive->GetDilateH();
  param->dilation_w_ = convg_primitive->GetDilateW();
  param->pad_u_ = convg_primitive->GetPadUp();
  param->pad_d_ = convg_primitive->GetPadDown();
  param->pad_l_ = convg_primitive->GetPadLeft();
  param->pad_r_ = convg_primitive->GetPadRight();
  param->group_ = convg_primitive->GetGroup();
  param->act_type_ = ActType_No;
  switch (convg_primitive->GetActivationType()) {
    case schema::ActivationType_RELU:
      param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      param->act_type_ = ActType_Relu6;
      break;
    default:
      break;
  }

  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateConvolutionGradInputParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }

  ConvParameter *param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc Param for conv grad filter failed.";
    return nullptr;
  }
  param->op_parameter_.type_ = primitive->Type();

  auto convg_primitive =
    reinterpret_cast<mindspore::lite::Conv2DGradInput *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  param->kernel_h_ = convg_primitive->GetKernelH();
  param->kernel_w_ = convg_primitive->GetKernelW();
  param->stride_h_ = convg_primitive->GetStrideH();
  param->stride_w_ = convg_primitive->GetStrideW();
  param->dilation_h_ = convg_primitive->GetDilateH();
  param->dilation_w_ = convg_primitive->GetDilateW();
  param->pad_u_ = convg_primitive->GetPadUp();
  param->pad_d_ = convg_primitive->GetPadDown();
  param->pad_l_ = convg_primitive->GetPadLeft();
  param->pad_r_ = convg_primitive->GetPadRight();
  param->group_ = convg_primitive->GetGroup();
  param->act_type_ = ActType_No;
  switch (convg_primitive->GetActivationType()) {
    case schema::ActivationType_RELU:
      param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      param->act_type_ = ActType_Relu6;
      break;
    default:
      break;
  }

  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateGroupConvolutionGradInputParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }

  ConvParameter *param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new Param for conv grad filter failed.";
    return nullptr;
  }
  param->op_parameter_.type_ = primitive->Type();

  auto convg_primitive =
    reinterpret_cast<mindspore::lite::GroupConv2DGradInput *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  param->kernel_h_ = convg_primitive->GetKernelH();
  param->kernel_w_ = convg_primitive->GetKernelW();
  param->stride_h_ = convg_primitive->GetStrideH();
  param->stride_w_ = convg_primitive->GetStrideW();
  param->dilation_h_ = convg_primitive->GetDilateH();
  param->dilation_w_ = convg_primitive->GetDilateW();
  param->pad_u_ = convg_primitive->GetPadUp();
  param->pad_d_ = convg_primitive->GetPadDown();
  param->pad_l_ = convg_primitive->GetPadLeft();
  param->pad_r_ = convg_primitive->GetPadRight();
  param->group_ = convg_primitive->GetGroup();
  param->act_type_ = ActType_No;
  switch (convg_primitive->GetActivationType()) {
    case schema::ActivationType_RELU:
      param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      param->act_type_ = ActType_Relu6;
      break;
    default:
      break;
  }

  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulatePowerGradParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }

  PowerParameter *power_param = reinterpret_cast<PowerParameter *>(malloc(sizeof(PowerParameter)));
  if (power_param == nullptr) {
    MS_LOG(ERROR) << "malloc PowerParameter failed.";
    return nullptr;
  }
  power_param->op_parameter_.type_ = primitive->Type();
  auto power = reinterpret_cast<mindspore::lite::PowerGrad *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  power_param->power_ = power->GetPower();
  power_param->scale_ = power->GetScale();
  power_param->shift_ = power->GetShift();
  return reinterpret_cast<OpParameter *>(power_param);
}

OpParameter *PopulateBiasGradParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }

  ArithmeticParameter *arithmetic_param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  arithmetic_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

OpParameter *PopulateBNGradParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }

  BNGradParameter *bnGrad_param = reinterpret_cast<BNGradParameter *>(malloc(sizeof(BNGradParameter)));
  if (bnGrad_param == nullptr) {
    MS_LOG(ERROR) << "malloc BNGradParameter failed.";
    return nullptr;
  }
  bnGrad_param->op_parameter_.type_ = primitive->Type();
  auto bngrad = reinterpret_cast<mindspore::lite::BNGrad *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  bnGrad_param->epsilon_ = bngrad->GetEps();
  bnGrad_param->momentum_ = bngrad->GetMomentum();
  return reinterpret_cast<OpParameter *>(bnGrad_param);
}

OpParameter *PopulateDropoutParameter(const mindspore::lite::PrimitiveC *primitive) {
  DropoutParameter *dropout_parameter = reinterpret_cast<DropoutParameter *>(malloc(sizeof(DropoutParameter)));
  if (dropout_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc Dropout Parameter failed.";
    return nullptr;
  }
  memset(dropout_parameter, 0, sizeof(DropoutParameter));
  dropout_parameter->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Dropout *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  dropout_parameter->ratio_ = param->GetRatio();
  if (dropout_parameter->ratio_ < 0.f || dropout_parameter->ratio_ > 1.f) {
    MS_LOG(ERROR) << "Dropout ratio must be between 0 to 1, got " << dropout_parameter->ratio_;
    free(dropout_parameter);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(dropout_parameter);
}

OpParameter *PopulateDropoutGradParameter(const mindspore::lite::PrimitiveC *primitive) {
  DropoutParameter *dropoutGrad_parameter = reinterpret_cast<DropoutParameter *>(malloc(sizeof(DropoutParameter)));
  if (dropoutGrad_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc Dropout Grad Parameter failed.";
    return nullptr;
  }
  memset(dropoutGrad_parameter, 0, sizeof(DropoutParameter));
  dropoutGrad_parameter->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::DropoutGrad *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  dropoutGrad_parameter->ratio_ = param->GetRatio();
  if (dropoutGrad_parameter->ratio_ < 0.f || dropoutGrad_parameter->ratio_ > 1.f) {
    MS_LOG(ERROR) << "Dropout Grad ratio must be between 0 to 1, got " << dropoutGrad_parameter->ratio_;
    free(dropoutGrad_parameter);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(dropoutGrad_parameter);
}

OpParameter *PopulateArithmeticGradParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticParameter *arithmetic_param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(arithmetic_param, 0, sizeof(ArithmeticParameter));
  arithmetic_param->op_parameter_.type_ = primitive->Type();
  arithmetic_param->broadcasting_ = ((lite::ArithmeticGrad *)primitive)->Broadcasting();
  arithmetic_param->ndim_ = ((lite::ArithmeticGrad *)primitive)->NDims();

  auto shape = ((lite::ArithmeticGrad *)primitive)->x1Shape();
  auto source = static_cast<int *>(shape.data());
  std::copy(source, source + shape.size(), arithmetic_param->in_shape0_);
  shape = ((lite::ArithmeticGrad *)primitive)->x2Shape();
  source = static_cast<int *>(shape.data());
  std::copy(source, source + shape.size(), arithmetic_param->in_shape1_);
  shape = ((lite::ArithmeticGrad *)primitive)->dyShape();
  source = static_cast<int *>(shape.data());
  std::copy(source, source + shape.size(), arithmetic_param->out_shape_);
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

void PopulateTrainParameters() {
  lite::Registry ApplyMomentumParameterRegistry(schema::PrimitiveType_ApplyMomentum, PopulateApplyMomentumParameter);
  lite::Registry BiasGradParameterRegistry(schema::PrimitiveType_BiasGrad, PopulateBiasGradParameter);
  lite::Registry SoftmaxCrossEntropyParameterRegistry(schema::PrimitiveType_SoftmaxCrossEntropy,
                                                      PopulateSoftmaxCrossEntropyParameter);
  lite::Registry SparseSoftmaxCrossEntropyParameterRegistry(schema::PrimitiveType_SparseSoftmaxCrossEntropy,
                                                            PopulateSparseSoftmaxCrossEntropyParameter);
  lite::Registry ActivationParameterRegistry(schema::PrimitiveType_ActivationGrad, PopulateActivationGradParameter);
  lite::Registry TupleGetItemParameterRegistry(schema::PrimitiveType_TupleGetItem, DefaultPopulateParameter);
  lite::Registry DependParameterRegistry(schema::PrimitiveType_Depend, DefaultPopulateParameter);
  lite::Registry Conv2DGradFilterParameterRegistry(schema::PrimitiveType_Conv2DGradFilter,
                                                   PopulateConvolutionGradFilterParameter);
  lite::Registry Conv2DGradInputParameterRegistry(schema::PrimitiveType_Conv2DGradInput,
                                                  PopulateConvolutionGradInputParameter);
  lite::Registry GroupConv2DGradInputParameterRegistry(schema::PrimitiveType_GroupConv2DGradInput,
                                                       PopulateGroupConvolutionGradInputParameter);
  lite::Registry PoolingParameterRegistry(schema::PrimitiveType_PoolingGrad, PopulatePoolingGradParameter);
  lite::Registry PowerGradParameterRegistry(schema::PrimitiveType_PowerGrad, PopulatePowerGradParameter);
  lite::Registry SgdParameterRegistry(schema::PrimitiveType_Sgd, PopulateSgdParameter);
  lite::Registry BNGradParameterRegistry(schema::PrimitiveType_BNGrad, PopulateBNGradParameter);
  lite::Registry AdamParameterRegistry(schema::PrimitiveType_Adam, PopulateAdamParameter);
  lite::Registry AssignParameterRegistry(schema::PrimitiveType_Assign, DefaultPopulateParameter);
  lite::Registry AssignAddParameterRegistry(schema::PrimitiveType_AssignAdd, DefaultPopulateParameter);
  lite::Registry BinaryCrossEntropyParameterRegistry(schema::PrimitiveType_BinaryCrossEntropy, PopulateBCEParameter);
  lite::Registry BinaryCrossEntropyGradParameterRegistry(schema::PrimitiveType_BinaryCrossEntropyGrad,
                                                         PopulateBCEGradParameter);
  lite::Registry OnesLikeParameterRegistry(schema::PrimitiveType_OnesLike, DefaultPopulateParameter);
  lite::Registry UnsortedSegmentSumParameterRegistry(schema::PrimitiveType_UnsortedSegmentSum,
                                                     DefaultPopulateParameter);
  lite::Registry DropoutParameterRegistry(schema::PrimitiveType_Dropout, PopulateDropoutParameter);
  lite::Registry DropGradParameterRegistry(schema::PrimitiveType_DropoutGrad, PopulateDropoutGradParameter);
  lite::Registry MaximumGradParameterRegistry(schema::PrimitiveType_MaximumGrad, PopulateArithmeticGradParameter);
  lite::Registry MinimumGradParameterRegistry(schema::PrimitiveType_MinimumGrad, PopulateArithmeticGradParameter);
  lite::Registry SmoothL1LossRegistry(schema::PrimitiveType_SmoothL1Loss, PopulateSmoothL1LossParameter);
  lite::Registry SmoothL1LossGradRegistry(schema::PrimitiveType_SmoothL1LossGrad, PopulateSmoothL1LossGradParameter);
  lite::Registry SigmoidCrossEntropyWithLogitsRegistry(schema::PrimitiveType_SigmoidCrossEntropyWithLogits,
                                                       DefaultPopulateParameter);
  lite::Registry SigmoidCrossEntropyWithLogitsGradRegistry(schema::PrimitiveType_SigmoidCrossEntropyWithLogitsGrad,
                                                           DefaultPopulateParameter);
}

}  // namespace mindspore::kernel
