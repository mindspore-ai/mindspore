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
#include "src/train/train_populate_parameter.h"
#include "src/ops/pooling_grad.h"
#include "nnacl/pooling_parameter.h"
#include "src/ops/softmax_cross_entropy.h"
#include "nnacl/fp32_grad/softmax_grad.h"
#include "src/ops/activation_grad.h"
#include "nnacl/fp32/activation.h"
#include "src/ops/conv2d_grad_filter.h"
#include "src/ops/conv2d_grad_input.h"
#include "nnacl/conv_parameter.h"
#include "src/ops/power_grad.h"
#include "nnacl/power_parameter.h"
#include "src/ops/bias_grad.h"
#include "nnacl/arithmetic_common.h"
#include "nnacl/fp32_grad/optimizer.h"
#include "src/ops/apply_momentum.h"
#include "src/ops/sgd.h"
#include "src/ops/bn_grad.h"
#include "nnacl/fp32_grad/batch_norm.h"

namespace mindspore::kernel {

OpParameter *DefaultPopulateParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }

  OpParameter *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new Param for primitive failed.";
    return nullptr;
  }

  param->type_ = primitive->Type();
  return param;
}

OpParameter *PopulateApplyMomentumParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  ApplyMomentumParameter *p = reinterpret_cast<ApplyMomentumParameter *>(malloc(sizeof(ApplyMomentumParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "new ApplyMomentumParameter failed.";
    return nullptr;
  }
  p->op_parameter_.type_ = primitive->Type();

  auto apply_momentum_primitive =
    reinterpret_cast<mindspore::lite::ApplyMomentum *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));

  p->grad_scale_ = apply_momentum_primitive->GetGradientScale();
  p->use_locking_ = apply_momentum_primitive->GetUseLocking();
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
    MS_LOG(ERROR) << "new SgdParameter failed.";
    return nullptr;
  }
  p->op_parameter_.type_ = primitive->Type();

  auto sgd_primitive = reinterpret_cast<mindspore::lite::Sgd *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));

  p->weight_decay_ = sgd_primitive->GetWeightDecay();
  p->dampening_ = sgd_primitive->GetDampening();
  p->use_nesterov_ = sgd_primitive->GetUseNesterov();

  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateSoftmaxCrossEntropyParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  SoftmaxCrossEntropyParameter *sce_param =
    reinterpret_cast<SoftmaxCrossEntropyParameter *>(malloc(sizeof(SoftmaxCrossEntropyParameter)));
  if (sce_param == nullptr) {
    MS_LOG(ERROR) << "new SoftmaxCrossEntropyParameter failed.";
    return nullptr;
  }
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
    MS_LOG(ERROR) << "new PoolingParameter failed.";
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
    MS_LOG(ERROR) << "new ActivationParameter failed.";
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
    MS_LOG(ERROR) << "new Param for conv grad filter failed.";
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
    MS_LOG(ERROR) << "new Param for conv grad filter failed.";
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

OpParameter *PopulatePowerGradParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }

  PowerParameter *power_param = reinterpret_cast<PowerParameter *>(malloc(sizeof(PowerParameter)));
  if (power_param == nullptr) {
    MS_LOG(ERROR) << "new PowerParameter failed.";
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
    MS_LOG(ERROR) << "new ArithmeticParameter failed.";
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
    MS_LOG(ERROR) << "new BNGradParameter failed.";
    return nullptr;
  }
  bnGrad_param->op_parameter_.type_ = primitive->Type();
  auto bngrad = reinterpret_cast<mindspore::lite::BNGrad *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  bnGrad_param->epsilon_ = bngrad->GetEps();
  bnGrad_param->momentum_ = 0.1;
  return reinterpret_cast<OpParameter *>(bnGrad_param);
}

void PopulateTrainParameters() {
  auto ppr = PopulateParameterRegistry::GetInstance();
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_ApplyMomentum, PopulateApplyMomentumParameter);
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_BiasGrad, PopulateBiasGradParameter);
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_SoftmaxCrossEntropy, PopulateSoftmaxCrossEntropyParameter);
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_ActivationGrad, PopulateActivationGradParameter);
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_TupleGetItem, DefaultPopulateParameter);
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_Depend, DefaultPopulateParameter);
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_BNGrad, DefaultPopulateParameter);
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_Conv2DGradFilter, PopulateConvolutionGradFilterParameter);
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_Conv2DGradInput, PopulateConvolutionGradInputParameter);
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_PoolingGrad, PopulatePoolingGradParameter);
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_PowerGrad, PopulatePowerGradParameter);
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_Sgd, PopulateSgdParameter);
  ppr->AddPopulateParameterFunc(schema::PrimitiveType_BNGrad, PopulateBNGradParameter);
}

}  // namespace mindspore::kernel
