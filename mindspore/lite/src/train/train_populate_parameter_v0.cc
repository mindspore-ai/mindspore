/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/train/train_populate_parameter_v0.h"
#include <vector>
#include "src/ops/populate/populate_register.h"
#include "schema/model_v0_generated.h"
#include "nnacl/pooling_parameter.h"
#include "nnacl/fp32_grad/softmax_grad.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/power_parameter.h"
#include "nnacl/arithmetic.h"
#include "nnacl/fp32_grad/optimizer.h"
#include "nnacl/fp32_grad/batch_norm.h"
#include "nnacl/fp32_grad/dropout_parameter.h"
#include "nnacl/fp32_grad/smooth_l1_loss.h"
#include "nnacl/infer/conv2d_grad_filter_infer.h"
#include "nnacl/infer/conv2d_grad_input_infer.h"
#include "nnacl/infer/group_conv2d_grad_input_infer.h"

namespace mindspore::kernel {
namespace {
OpParameter *DefaultPopulateParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);

  OpParameter *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc Param for primitive failed.";
    return nullptr;
  }
  auto type = prim->value_type();
  switch (prim->value_type()) {
    case schema::v0::PrimitiveType_Depend:
      param->type_ = schema::PrimitiveType_Depend;
      break;
    case schema::v0::PrimitiveType_Assign:
      param->type_ = schema::PrimitiveType_Assign;
      break;
    case schema::v0::PrimitiveType_AssignAdd:
      param->type_ = schema::PrimitiveType_AssignAdd;
      break;
    case schema::v0::PrimitiveType_OnesLike:
      param->type_ = schema::PrimitiveType_OnesLike;
      break;
    case schema::v0::PrimitiveType_UnsortedSegmentSum:
      param->type_ = schema::PrimitiveType_UnsortedSegmentSum;
      break;
    case schema::v0::PrimitiveType_SigmoidCrossEntropyWithLogits:
      param->type_ = schema::PrimitiveType_SigmoidCrossEntropyWithLogits;
      break;
    case schema::v0::PrimitiveType_SigmoidCrossEntropyWithLogitsGrad:
      param->type_ = schema::PrimitiveType_SigmoidCrossEntropyWithLogitsGrad;
      break;
    case schema::v0::PrimitiveType_AddGrad:
      param->type_ = schema::PrimitiveType_AddGrad;
      break;
    case schema::v0::PrimitiveType_SubGrad:
      param->type_ = schema::PrimitiveType_SubGrad;
      break;
    case schema::v0::PrimitiveType_MulGrad:
      param->type_ = schema::PrimitiveType_MulGrad;
      break;
    case schema::v0::PrimitiveType_DivGrad:
      param->type_ = schema::PrimitiveType_DivGrad;
      break;
    default:
      MS_LOG(ERROR) << "unsupported type: " << schema::v0::EnumNamePrimitiveType(type);
      free(param);
      return nullptr;
  }

  return param;
}

OpParameter *PopulateSmoothL1LossParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);
  SmoothL1LossParameter *p = reinterpret_cast<SmoothL1LossParameter *>(malloc(sizeof(SmoothL1LossParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "malloc SmoothL1LossParameter failed.";
    return nullptr;
  }
  p->op_parameter_.type_ = schema::PrimitiveType_SmoothL1Loss;

  auto smoothL1Loss_prim = prim->value_as_SmoothL1Loss();

  p->beta_ = smoothL1Loss_prim->beta();
  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateSmoothL1LossGradParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);
  SmoothL1LossParameter *p = reinterpret_cast<SmoothL1LossParameter *>(malloc(sizeof(SmoothL1LossParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "malloc SmoothL1LossParameter failed.";
    return nullptr;
  }
  p->op_parameter_.type_ = schema::PrimitiveType_SmoothL1LossGrad;

  auto smoothL1LossGrad_prim = prim->value_as_SmoothL1LossGrad();

  p->beta_ = smoothL1LossGrad_prim->beta();
  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateApplyMomentumParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);
  ApplyMomentumParameter *p = reinterpret_cast<ApplyMomentumParameter *>(malloc(sizeof(ApplyMomentumParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "malloc ApplyMomentumParameter failed.";
    return nullptr;
  }
  p->op_parameter_.type_ = schema::PrimitiveType_ApplyMomentum;

  auto applyMomentum_prim = prim->value_as_ApplyMomentum();

  p->grad_scale_ = applyMomentum_prim->gradientScale();
  p->use_nesterov_ = applyMomentum_prim->useNesterov();

  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateBCEParameter(const void *primitive) {
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);
  int32_t *reduction = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  if (reduction == nullptr) {
    MS_LOG(ERROR) << "malloc reduction failed.";
    return nullptr;
  }
  auto bCE_prim = prim->value_as_BinaryCrossEntropy();
  *reduction = bCE_prim->reduction();
  return reinterpret_cast<OpParameter *>(reduction);
}

OpParameter *PopulateBCEGradParameter(const void *primitive) {
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);
  int32_t *reduction = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  if (reduction == nullptr) {
    MS_LOG(ERROR) << "malloc reduction failed.";
    return nullptr;
  }
  auto bCEGrad_prim = prim->value_as_BinaryCrossEntropyGrad();

  *reduction = bCEGrad_prim->reduction();
  return reinterpret_cast<OpParameter *>(reduction);
}

OpParameter *PopulateAdamParameter(const void *primitive) {
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  AdamParameter *p = reinterpret_cast<AdamParameter *>(malloc(sizeof(AdamParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "new AdamParameter failed.";
    return nullptr;
  }
  p->op_parameter_.type_ = schema::PrimitiveType_Adam;

  auto adam_prim = prim->value_as_Adam();

  p->use_nesterov_ = adam_prim->useNesterov();
  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateSgdParameter(const void *primitive) {
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  SgdParameter *p = reinterpret_cast<SgdParameter *>(malloc(sizeof(SgdParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "malloc SgdParameter failed.";
    return nullptr;
  }
  p->op_parameter_.type_ = schema::PrimitiveType_SGD;

  auto sgd_prim = prim->value_as_Sgd();

  p->weight_decay_ = sgd_prim->weightDecay();
  p->dampening_ = sgd_prim->dampening();
  p->use_nesterov_ = sgd_prim->useNesterov();

  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateSparseSoftmaxCrossEntropyParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);
  SoftmaxCrossEntropyParameter *sce_param =
    reinterpret_cast<SoftmaxCrossEntropyParameter *>(malloc(sizeof(SoftmaxCrossEntropyParameter)));
  if (sce_param == nullptr) {
    MS_LOG(ERROR) << "malloc SoftmaxCrossEntropyParameter failed.";
    return nullptr;
  }
  auto sparseSoftmaxCrossEntropy_prim = prim->value_as_SparseSoftmaxCrossEntropy();

  sce_param->is_grad_ = sparseSoftmaxCrossEntropy_prim->isGrad();

  sce_param->op_parameter_.type_ = schema::PrimitiveType_SparseSoftmaxCrossEntropyWithLogits;
  return reinterpret_cast<OpParameter *>(sce_param);
}

OpParameter *PopulateSoftmaxCrossEntropyParameter(const void *primitive) {
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
  sce_param->is_grad_ = 0;
  sce_param->op_parameter_.type_ = schema::PrimitiveType_SoftmaxCrossEntropyWithLogits;
  return reinterpret_cast<OpParameter *>(sce_param);
}

OpParameter *PopulatePoolingGradParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);
  PoolingParameter *pooling_param = reinterpret_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  if (pooling_param == nullptr) {
    MS_LOG(ERROR) << "malloc PoolingParameter failed.";
    return nullptr;
  }

  auto poolingGrad_prim = prim->value_as_PoolingGrad();

  pooling_param->global_ = poolingGrad_prim->global();
  pooling_param->window_w_ = poolingGrad_prim->windowW();
  pooling_param->window_h_ = poolingGrad_prim->windowH();

  pooling_param->pad_u_ = poolingGrad_prim->padUp();
  pooling_param->pad_d_ = poolingGrad_prim->padDown();
  pooling_param->pad_l_ = poolingGrad_prim->padLeft();
  pooling_param->pad_r_ = poolingGrad_prim->padRight();
  pooling_param->stride_w_ = poolingGrad_prim->strideW();
  pooling_param->stride_h_ = poolingGrad_prim->strideH();

  pooling_param->pool_mode_ = PoolMode_No;
  pooling_param->round_mode_ = RoundMode_No;

  switch (poolingGrad_prim->poolingMode()) {
    case schema::v0::PoolMode_MAX_POOLING:
      pooling_param->pool_mode_ = PoolMode_MaxPool;
      pooling_param->op_parameter_.type_ = schema::PrimitiveType_MaxPoolGrad;
      break;
    case schema::v0::PoolMode_MEAN_POOLING:
      pooling_param->pool_mode_ = PoolMode_AvgPool;
      pooling_param->op_parameter_.type_ = schema::PrimitiveType_AvgPoolGrad;
      break;
    default:
      MS_LOG(ERROR) << "unknown pooling mode: " << poolingGrad_prim->poolingMode();
      free(pooling_param);
      return nullptr;
  }

  switch (poolingGrad_prim->roundMode()) {
    case schema::v0::RoundMode_FLOOR:
      pooling_param->round_mode_ = RoundMode_Floor;
      break;
    case schema::v0::RoundMode_CEIL:
      pooling_param->round_mode_ = RoundMode_Ceil;
      break;
    default:
      break;
  }
  return reinterpret_cast<OpParameter *>(pooling_param);
}

OpParameter *PopulateActivationGradParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);

  ActivationParameter *act_param = reinterpret_cast<ActivationParameter *>(malloc(sizeof(ActivationParameter)));
  if (act_param == nullptr) {
    MS_LOG(ERROR) << "malloc ActivationParameter failed.";
    return nullptr;
  }
  act_param->op_parameter_.type_ = schema::PrimitiveType_ActivationGrad;
  auto activationGrad_prim = prim->value_as_ActivationGrad();

  act_param->type_ = static_cast<int>(activationGrad_prim->type());
  act_param->alpha_ = activationGrad_prim->alpha();
  return reinterpret_cast<OpParameter *>(act_param);
}

OpParameter *PopulateConvolutionGradFilterParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);

  ConvParameter *param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc Param for conv grad filter failed.";
    return nullptr;
  }
  param->op_parameter_.type_ = schema::PrimitiveType_Conv2DBackpropFilterFusion;

  auto convolutionGradFilter_prim = prim->value_as_Conv2DGradFilter();
  auto fb_vector = convolutionGradFilter_prim->filter_shape();
  auto filter_shape = std::vector<int>(fb_vector->begin(), fb_vector->end());
  if (filter_shape.size() > MAX_SHAPE_SIZE) {
    free(param);
    MS_LOG(ERROR) << "ConvolutionGradFilter filter shape too big.";
    return nullptr;
  }
  param->kernel_h_ = convolutionGradFilter_prim->kernelH();
  param->kernel_w_ = convolutionGradFilter_prim->kernelW();
  param->stride_h_ = convolutionGradFilter_prim->strideH();
  param->stride_w_ = convolutionGradFilter_prim->strideW();
  param->dilation_h_ = convolutionGradFilter_prim->dilateH();
  param->dilation_w_ = convolutionGradFilter_prim->dilateW();
  param->pad_u_ = convolutionGradFilter_prim->padUp();
  param->pad_d_ = convolutionGradFilter_prim->padDown();
  param->pad_l_ = convolutionGradFilter_prim->padLeft();
  param->pad_r_ = convolutionGradFilter_prim->padRight();
  param->group_ = convolutionGradFilter_prim->group();
  param->act_type_ = ActType_No;
  switch (convolutionGradFilter_prim->activationType()) {
    case schema::v0::ActivationType_RELU:
      param->act_type_ = ActType_Relu;
      break;
    case schema::v0::ActivationType_RELU6:
      param->act_type_ = ActType_Relu6;
      break;
    default:
      break;
  }

  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateConvolutionGradInputParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);

  ConvParameter *param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc Param for conv grad filter failed.";
    return nullptr;
  }
  param->op_parameter_.type_ = schema::PrimitiveType_Conv2DBackpropInputFusion;

  auto convolutionGradInput_prim = prim->value_as_Conv2DGradInput();
  auto fb_vector = convolutionGradInput_prim->input_shape();
  auto filter_shape = std::vector<int>(fb_vector->begin(), fb_vector->end());
  if (filter_shape.size() > MAX_SHAPE_SIZE) {
    free(param);
    MS_LOG(ERROR) << "ConvolutionGradInput input shape too big.";
    return nullptr;
  }
  param->kernel_h_ = convolutionGradInput_prim->kernelH();
  param->kernel_w_ = convolutionGradInput_prim->kernelW();
  param->stride_h_ = convolutionGradInput_prim->strideH();
  param->stride_w_ = convolutionGradInput_prim->strideW();
  param->dilation_h_ = convolutionGradInput_prim->dilateH();
  param->dilation_w_ = convolutionGradInput_prim->dilateW();
  param->pad_u_ = convolutionGradInput_prim->padUp();
  param->pad_d_ = convolutionGradInput_prim->padDown();
  param->pad_l_ = convolutionGradInput_prim->padLeft();
  param->pad_r_ = convolutionGradInput_prim->padRight();
  param->group_ = convolutionGradInput_prim->group();
  param->act_type_ = ActType_No;
  switch (convolutionGradInput_prim->activationType()) {
    case schema::v0::ActivationType_RELU:
      param->act_type_ = ActType_Relu;
      break;
    case schema::v0::ActivationType_RELU6:
      param->act_type_ = ActType_Relu6;
      break;
    default:
      break;
  }

  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateGroupConvolutionGradInputParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);

  ConvParameter *param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new Param for conv grad filter failed.";
    return nullptr;
  }
  param->op_parameter_.type_ = schema::PrimitiveType_Conv2DBackpropInputFusion;

  auto groupConvolutionGradInput_prim = prim->value_as_GroupConv2DGradInput();
  auto fb_vector = groupConvolutionGradInput_prim->input_shape();
  auto filter_shape = std::vector<int>(fb_vector->begin(), fb_vector->end());
  if (filter_shape.size() > MAX_SHAPE_SIZE) {
    free(param);
    MS_LOG(ERROR) << "GroupConvolutionGradInput input shape too big.";
    return nullptr;
  }
  param->kernel_h_ = groupConvolutionGradInput_prim->kernelH();
  param->kernel_w_ = groupConvolutionGradInput_prim->kernelW();
  param->stride_h_ = groupConvolutionGradInput_prim->strideH();
  param->stride_w_ = groupConvolutionGradInput_prim->strideW();
  param->dilation_h_ = groupConvolutionGradInput_prim->dilateH();
  param->dilation_w_ = groupConvolutionGradInput_prim->dilateW();
  param->pad_u_ = groupConvolutionGradInput_prim->padUp();
  param->pad_d_ = groupConvolutionGradInput_prim->padDown();
  param->pad_l_ = groupConvolutionGradInput_prim->padLeft();
  param->pad_r_ = groupConvolutionGradInput_prim->padRight();
  param->group_ = groupConvolutionGradInput_prim->group();
  param->act_type_ = ActType_No;
  switch (groupConvolutionGradInput_prim->activationType()) {
    case schema::v0::ActivationType_RELU:
      param->act_type_ = ActType_Relu;
      break;
    case schema::v0::ActivationType_RELU6:
      param->act_type_ = ActType_Relu6;
      break;
    default:
      break;
  }

  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulatePowerGradParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);

  PowerParameter *power_param = reinterpret_cast<PowerParameter *>(malloc(sizeof(PowerParameter)));
  if (power_param == nullptr) {
    MS_LOG(ERROR) << "malloc PowerParameter failed.";
    return nullptr;
  }
  power_param->op_parameter_.type_ = schema::PrimitiveType_PowerGrad;
  auto powerGrad_prim = prim->value_as_PowerGrad();

  power_param->power_ = powerGrad_prim->power();
  power_param->scale_ = powerGrad_prim->scale();
  power_param->shift_ = powerGrad_prim->shift();
  return reinterpret_cast<OpParameter *>(power_param);
}

OpParameter *PopulateBiasGradParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }

  ArithmeticParameter *arithmetic_param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  arithmetic_param->op_parameter_.type_ = schema::PrimitiveType_BiasAddGrad;
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

OpParameter *PopulateBNGradParameter(const void *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr when populating parameter for op.";
    return nullptr;
  }
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);

  BNGradParameter *bnGrad_param = reinterpret_cast<BNGradParameter *>(malloc(sizeof(BNGradParameter)));
  if (bnGrad_param == nullptr) {
    MS_LOG(ERROR) << "malloc BNGradParameter failed.";
    return nullptr;
  }
  bnGrad_param->op_parameter_.type_ = schema::PrimitiveType_BatchNormGrad;
  auto bNGrad_prim = prim->value_as_BNGrad();

  bnGrad_param->epsilon_ = bNGrad_prim->eps();
  return reinterpret_cast<OpParameter *>(bnGrad_param);
}

OpParameter *PopulateDropoutParameter(const void *primitive) {
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);
  DropoutParameter *dropout_parameter = reinterpret_cast<DropoutParameter *>(malloc(sizeof(DropoutParameter)));
  if (dropout_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc Dropout Parameter failed.";
    return nullptr;
  }
  memset(dropout_parameter, 0, sizeof(DropoutParameter));
  dropout_parameter->op_parameter_.type_ = schema::PrimitiveType_Dropout;
  auto dropout_prim = prim->value_as_Dropout();

  dropout_parameter->ratio_ = dropout_prim->ratio();
  if (dropout_parameter->ratio_ < 0.f || dropout_parameter->ratio_ > 1.f) {
    MS_LOG(ERROR) << "Dropout ratio must be between 0 to 1, got " << dropout_parameter->ratio_;
    free(dropout_parameter);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(dropout_parameter);
}

OpParameter *PopulateDropoutGradParameter(const void *primitive) {
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);
  DropoutParameter *dropoutGrad_parameter = reinterpret_cast<DropoutParameter *>(malloc(sizeof(DropoutParameter)));
  if (dropoutGrad_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc Dropout Grad Parameter failed.";
    return nullptr;
  }
  memset(dropoutGrad_parameter, 0, sizeof(DropoutParameter));
  dropoutGrad_parameter->op_parameter_.type_ = schema::PrimitiveType_DropoutGrad;
  auto dropoutGrad_prim = prim->value_as_DropoutGrad();

  dropoutGrad_parameter->ratio_ = dropoutGrad_prim->ratio();
  if (dropoutGrad_parameter->ratio_ < 0.f || dropoutGrad_parameter->ratio_ > 1.f) {
    MS_LOG(ERROR) << "Dropout Grad ratio must be between 0 to 1, got " << dropoutGrad_parameter->ratio_;
    free(dropoutGrad_parameter);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(dropoutGrad_parameter);
}

OpParameter *PopulateArithmeticGradParameter(const void *primitive) {
  ArithmeticParameter *arithmetic_param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(arithmetic_param, 0, sizeof(ArithmeticParameter));
  auto *prim = static_cast<const schema::v0::Primitive *>(primitive);
  if (prim->value_type() == schema::v0::PrimitiveType_MaximumGrad) {
    arithmetic_param->op_parameter_.type_ = schema::PrimitiveType_MaximumGrad;
  } else if (prim->value_type() == schema::v0::PrimitiveType_MinimumGrad) {
    arithmetic_param->op_parameter_.type_ = schema::PrimitiveType_MinimumGrad;
  } else {
    MS_LOG(ERROR) << "unsupported type: " << schema::v0::EnumNamePrimitiveType(prim->value_type());
    free(arithmetic_param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

}  // namespace

void PopulateTrainV0Parameters() {
  lite::Registry g_applyMomentumV0ParameterRegistry(schema::v0::PrimitiveType_ApplyMomentum,
                                                    PopulateApplyMomentumParameter, mindspore::lite::SCHEMA_V0);
  lite::Registry g_addGradV0ParameterRegistry(schema::v0::PrimitiveType_AddGrad, DefaultPopulateParameter,
                                              mindspore::lite::SCHEMA_V0);
  lite::Registry g_subGradV0ParameterRegistry(schema::v0::PrimitiveType_SubGrad, DefaultPopulateParameter,
                                              mindspore::lite::SCHEMA_V0);
  lite::Registry g_mulGradV0ParameterRegistry(schema::v0::PrimitiveType_MulGrad, DefaultPopulateParameter,
                                              mindspore::lite::SCHEMA_V0);
  lite::Registry g_divGradV0ParameterRegistry(schema::v0::PrimitiveType_DivGrad, DefaultPopulateParameter,
                                              mindspore::lite::SCHEMA_V0);
  lite::Registry g_biasGradV0ParameterRegistry(schema::v0::PrimitiveType_BiasGrad, PopulateBiasGradParameter,
                                               mindspore::lite::SCHEMA_V0);
  lite::Registry g_softmaxCrossEntropyV0ParameterRegistry(
    schema::v0::PrimitiveType_SoftmaxCrossEntropy, PopulateSoftmaxCrossEntropyParameter, mindspore::lite::SCHEMA_V0);
  lite::Registry g_sparseSoftmaxCrossEntropyV0ParameterRegistry(schema::v0::PrimitiveType_SparseSoftmaxCrossEntropy,
                                                                PopulateSparseSoftmaxCrossEntropyParameter,
                                                                mindspore::lite::SCHEMA_V0);
  lite::Registry g_activationV0ParameterRegistry(schema::v0::PrimitiveType_ActivationGrad,
                                                 PopulateActivationGradParameter, mindspore::lite::SCHEMA_V0);
  lite::Registry g_tupleGetItemV0ParameterRegistry(schema::v0::PrimitiveType_TupleGetItem, DefaultPopulateParameter,
                                                   mindspore::lite::SCHEMA_V0);
  lite::Registry g_dependV0ParameterRegistry(schema::v0::PrimitiveType_Depend, DefaultPopulateParameter,
                                             mindspore::lite::SCHEMA_V0);
  lite::Registry g_conv2DGradFilterV0ParameterRegistry(
    schema::v0::PrimitiveType_Conv2DGradFilter, PopulateConvolutionGradFilterParameter, mindspore::lite::SCHEMA_V0);
  lite::Registry g_conv2DGradInputV0ParameterRegistry(
    schema::v0::PrimitiveType_Conv2DGradInput, PopulateConvolutionGradInputParameter, mindspore::lite::SCHEMA_V0);
  lite::Registry g_groupConv2DGradInputV0ParameterRegistry(schema::v0::PrimitiveType_GroupConv2DGradInput,
                                                           PopulateGroupConvolutionGradInputParameter,
                                                           mindspore::lite::SCHEMA_V0);
  lite::Registry g_poolingV0ParameterRegistry(schema::v0::PrimitiveType_PoolingGrad, PopulatePoolingGradParameter,
                                              mindspore::lite::SCHEMA_V0);
  lite::Registry g_powerGradV0ParameterRegistry(schema::v0::PrimitiveType_PowerGrad, PopulatePowerGradParameter,
                                                mindspore::lite::SCHEMA_V0);
  lite::Registry g_sgdV0ParameterRegistry(schema::v0::PrimitiveType_Sgd, PopulateSgdParameter,
                                          mindspore::lite::SCHEMA_V0);
  lite::Registry g_bNGradV0ParameterRegistry(schema::v0::PrimitiveType_BNGrad, PopulateBNGradParameter,
                                             mindspore::lite::SCHEMA_V0);
  lite::Registry g_adamV0ParameterRegistry(schema::v0::PrimitiveType_Adam, PopulateAdamParameter,
                                           mindspore::lite::SCHEMA_V0);
  lite::Registry g_assignV0ParameterRegistry(schema::v0::PrimitiveType_Assign, DefaultPopulateParameter,
                                             mindspore::lite::SCHEMA_V0);
  lite::Registry g_assignAddV0ParameterRegistry(schema::v0::PrimitiveType_AssignAdd, DefaultPopulateParameter,
                                                mindspore::lite::SCHEMA_V0);
  lite::Registry g_binaryCrossEntropyV0ParameterRegistry(schema::v0::PrimitiveType_BinaryCrossEntropy,
                                                         PopulateBCEParameter, mindspore::lite::SCHEMA_V0);
  lite::Registry g_binaryCrossEntropyGradV0ParameterRegistry(schema::v0::PrimitiveType_BinaryCrossEntropyGrad,
                                                             PopulateBCEGradParameter, mindspore::lite::SCHEMA_V0);
  lite::Registry g_onesLikeV0ParameterRegistry(schema::v0::PrimitiveType_OnesLike, DefaultPopulateParameter,
                                               mindspore::lite::SCHEMA_V0);
  lite::Registry g_unsortedSegmentSumV0ParameterRegistry(schema::v0::PrimitiveType_UnsortedSegmentSum,
                                                         DefaultPopulateParameter, mindspore::lite::SCHEMA_V0);
  lite::Registry g_dropoutV0ParameterRegistry(schema::v0::PrimitiveType_Dropout, PopulateDropoutParameter,
                                              mindspore::lite::SCHEMA_V0);
  lite::Registry g_dropGradV0ParameterRegistry(schema::v0::PrimitiveType_DropoutGrad, PopulateDropoutGradParameter,
                                               mindspore::lite::SCHEMA_V0);
  lite::Registry g_maximumGradV0ParameterRegistry(schema::v0::PrimitiveType_MaximumGrad,
                                                  PopulateArithmeticGradParameter, mindspore::lite::SCHEMA_V0);
  lite::Registry g_minimumGradV0ParameterRegistry(schema::v0::PrimitiveType_MinimumGrad,
                                                  PopulateArithmeticGradParameter, mindspore::lite::SCHEMA_V0);
  lite::Registry g_smoothL1LossRegistry(schema::v0::PrimitiveType_SmoothL1Loss, PopulateSmoothL1LossParameter,
                                        mindspore::lite::SCHEMA_V0);
  lite::Registry g_smoothL1LossGradRegistry(schema::v0::PrimitiveType_SmoothL1LossGrad,
                                            PopulateSmoothL1LossGradParameter, mindspore::lite::SCHEMA_V0);
  lite::Registry g_sigmoidCrossEntropyWithLogitsRegistry(schema::v0::PrimitiveType_SigmoidCrossEntropyWithLogits,
                                                         DefaultPopulateParameter, mindspore::lite::SCHEMA_V0);
  lite::Registry g_sigmoidCrossEntropyWithLogitsGradRegistry(
    schema::v0::PrimitiveType_SigmoidCrossEntropyWithLogitsGrad, DefaultPopulateParameter, mindspore::lite::SCHEMA_V0);
}

}  // namespace mindspore::kernel
