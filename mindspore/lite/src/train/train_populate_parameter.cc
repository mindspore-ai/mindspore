/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "src/common/ops/populate/populate_register.h"
#include "src/common/ops/populate/default_populate.h"
#include "nnacl/strided_slice_parameter.h"
#include "nnacl/arithmetic.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/pooling_parameter.h"
#include "nnacl/power_parameter.h"
#include "nnacl/activation_parameter.h"
#include "nnacl/fp32_grad/softmax_crossentropy_parameter.h"
#include "nnacl/fp32_grad/optimizer.h"
#include "nnacl/fp32_grad/batch_norm_parameter.h"
#include "nnacl/fp32_grad/dropout_parameter.h"
#include "nnacl/fp32_grad/smooth_l1_loss.h"
#include "nnacl/fp32_grad/resize_grad_parameter.h"
#include "nnacl/fp32_grad/lstm_grad_fp32.h"
#include "nnacl/fp32_grad/binary_cross_entropy.h"
#include "nnacl/fp32_grad/binary_cross_entropy_grad.h"

using mindspore::lite::Registry;

namespace mindspore {
namespace kernel {
namespace {
constexpr int kInputIndexOne = 1;
constexpr int kInputIndexTwo = 2;
constexpr int kInputIndexThree = 3;
}  // namespace
OpParameter *PopulateSmoothL1LossParameter(const void *prim) {
  SmoothL1LossParameter *p = reinterpret_cast<SmoothL1LossParameter *>(malloc(sizeof(SmoothL1LossParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "malloc SmoothL1LossParameter failed.";
    return nullptr;
  }
  memset(p, 0, sizeof(SmoothL1LossParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_SmoothL1Loss();
  MS_ASSERT(value != nullptr);
  p->op_parameter_.type_ = primitive->value_type();
  p->beta_ = value->beta();
  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateSmoothL1LossGradParameter(const void *prim) {
  SmoothL1LossParameter *p = reinterpret_cast<SmoothL1LossParameter *>(malloc(sizeof(SmoothL1LossParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "malloc SmoothL1LossParameter failed.";
    return nullptr;
  }
  memset(p, 0, sizeof(SmoothL1LossParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_SmoothL1LossGrad();
  MS_ASSERT(value != nullptr);
  p->op_parameter_.type_ = primitive->value_type();
  p->beta_ = value->beta();
  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateApplyMomentumParameter(const void *prim) {
  ApplyMomentumParameter *p = reinterpret_cast<ApplyMomentumParameter *>(malloc(sizeof(ApplyMomentumParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "malloc ApplyMomentumParameter failed.";
    return nullptr;
  }
  memset(p, 0, sizeof(ApplyMomentumParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_ApplyMomentum();
  MS_ASSERT(value != nullptr);
  p->op_parameter_.type_ = primitive->value_type();
  p->grad_scale_ = value->gradient_scale();
  p->use_nesterov_ = value->use_nesterov();
  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateBCEParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_BinaryCrossEntropy();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<BinaryCrossEntropyParameter *>(malloc(sizeof(BinaryCrossEntropyParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc BinaryCrossEntropy Parameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(BinaryCrossEntropyParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->reduction = value->reduction();
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateBCEGradParameter(const void *prim) {
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_BinaryCrossEntropyGrad();
  if (value == nullptr) {
    MS_LOG(ERROR) << "param is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<BinaryCrossEntropyGradParameter *>(malloc(sizeof(BinaryCrossEntropyGradParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc BinaryCrossEntropyGrad Parameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(BinaryCrossEntropyGradParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->reduction = value->reduction();
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateAdamParameter(const void *prim) {
  AdamParameter *p = reinterpret_cast<AdamParameter *>(malloc(sizeof(AdamParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "new AdamParameter failed.";
    return nullptr;
  }
  memset(p, 0, sizeof(AdamParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_Adam();
  MS_ASSERT(value != nullptr);
  p->op_parameter_.type_ = primitive->value_type();
  p->use_nesterov_ = value->use_nesterov();
  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateSgdParameter(const void *prim) {
  SgdParameter *p = reinterpret_cast<SgdParameter *>(malloc(sizeof(SgdParameter)));
  if (p == nullptr) {
    MS_LOG(ERROR) << "malloc SgdParameter failed.";
    return nullptr;
  }
  memset(p, 0, sizeof(SgdParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_SGD();
  MS_ASSERT(value != nullptr);
  p->op_parameter_.type_ = primitive->value_type();
  p->weight_decay_ = value->weight_decay();
  p->dampening_ = value->dampening();
  p->use_nesterov_ = value->nesterov();

  return reinterpret_cast<OpParameter *>(p);
}

OpParameter *PopulateSparseSoftmaxCrossEntropyWithLogitsParameter(const void *prim) {
  SoftmaxCrossEntropyParameter *sce_param =
    reinterpret_cast<SoftmaxCrossEntropyParameter *>(malloc(sizeof(SoftmaxCrossEntropyParameter)));
  if (sce_param == nullptr) {
    MS_LOG(ERROR) << "malloc SoftmaxCrossEntropyParameter failed.";
    return nullptr;
  }
  memset(sce_param, 0, sizeof(SoftmaxCrossEntropyParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_SparseSoftmaxCrossEntropyWithLogits();
  MS_ASSERT(value != nullptr);
  sce_param->op_parameter_.type_ = primitive->value_type();
  sce_param->is_grad_ = value->is_grad();
  return reinterpret_cast<OpParameter *>(sce_param);
}

OpParameter *PopulateSoftmaxCrossEntropyParameter(const void *prim) {
  SoftmaxCrossEntropyParameter *sce_param =
    reinterpret_cast<SoftmaxCrossEntropyParameter *>(malloc(sizeof(SoftmaxCrossEntropyParameter)));
  if (sce_param == nullptr) {
    MS_LOG(ERROR) << "malloc SoftmaxCrossEntropyParameter failed.";
    return nullptr;
  }
  memset(sce_param, 0, sizeof(SoftmaxCrossEntropyParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  sce_param->op_parameter_.type_ = primitive->value_type();
  sce_param->is_grad_ = false;
  return reinterpret_cast<OpParameter *>(sce_param);
}

OpParameter *PopulateMaxPoolGradParameter(const void *prim) {
  PoolingParameter *pooling_param = reinterpret_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  if (pooling_param == nullptr) {
    MS_LOG(ERROR) << "malloc PoolingParameter failed.";
    return nullptr;
  }
  memset(pooling_param, 0, sizeof(PoolingParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_MaxPoolGrad();
  MS_ASSERT(value != nullptr);
  pooling_param->op_parameter_.type_ = primitive->value_type();

  pooling_param->global_ = false;
  pooling_param->window_w_ = static_cast<int>(value->kernel_size()->Get(1));
  pooling_param->window_h_ = static_cast<int>(value->kernel_size()->Get(0));

  pooling_param->pad_u_ = 0;
  pooling_param->pad_d_ = 0;
  pooling_param->pad_l_ = 0;
  pooling_param->pad_r_ = 0;
  pooling_param->stride_w_ = static_cast<int>(value->strides()->Get(1));
  pooling_param->stride_h_ = static_cast<int>(value->strides()->Get(0));
  pooling_param->round_mode_ = RoundMode_No;
  pooling_param->pool_mode_ = PoolMode_MaxPool;
  switch (value->pad_mode()) {
    case schema::PadMode_SAME:
      pooling_param->pad_mode_ = Pad_same;
      break;
    case schema::PadMode_VALID:
      pooling_param->pad_mode_ = Pad_valid;
      break;
    default:
      pooling_param->pad_mode_ = Pad_pad;
      break;
  }

  return reinterpret_cast<OpParameter *>(pooling_param);
}

OpParameter *PopulateAvgPoolGradParameter(const void *prim) {
  PoolingParameter *pooling_param = reinterpret_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  if (pooling_param == nullptr) {
    MS_LOG(ERROR) << "malloc PoolingParameter failed.";
    return nullptr;
  }
  memset(pooling_param, 0, sizeof(PoolingParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_AvgPoolGrad();
  MS_ASSERT(value != nullptr);
  pooling_param->op_parameter_.type_ = primitive->value_type();

  pooling_param->global_ = false;
  pooling_param->window_w_ = static_cast<int>(value->kernel_size()->Get(1));
  pooling_param->window_h_ = static_cast<int>(value->kernel_size()->Get(0));

  pooling_param->pad_u_ = 0;
  pooling_param->pad_d_ = 0;
  pooling_param->pad_l_ = 0;
  pooling_param->pad_r_ = 0;
  pooling_param->stride_w_ = static_cast<int>(value->strides()->Get(1));
  pooling_param->stride_h_ = static_cast<int>(value->strides()->Get(0));

  switch (value->pad_mode()) {
    case schema::PadMode_SAME:
      pooling_param->pad_mode_ = Pad_same;
      break;
    case schema::PadMode_VALID:
      pooling_param->pad_mode_ = Pad_valid;
      break;
    default:
      pooling_param->pad_mode_ = Pad_pad;
      break;
  }
  pooling_param->round_mode_ = RoundMode_No;
  pooling_param->pool_mode_ = PoolMode_AvgPool;
  return reinterpret_cast<OpParameter *>(pooling_param);
}

OpParameter *PopulateActivationGradParameter(const void *prim) {
  ActivationParameter *act_param = reinterpret_cast<ActivationParameter *>(malloc(sizeof(ActivationParameter)));
  if (act_param == nullptr) {
    MS_LOG(ERROR) << "malloc ActivationParameter failed.";
    return nullptr;
  }
  memset(act_param, 0, sizeof(ActivationParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_ActivationGrad();
  MS_ASSERT(value != nullptr);
  act_param->op_parameter_.type_ = primitive->value_type();
  act_param->type_ = static_cast<int>(value->activation_type());
  act_param->alpha_ = value->alpha();
  return reinterpret_cast<OpParameter *>(act_param);
}

OpParameter *PopulateConvolutionGradFilterParameter(const void *prim) {
  ConvParameter *param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc Param for conv grad filter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ConvParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_Conv2DBackpropFilterFusion();
  MS_ASSERT(value != nullptr);
  param->op_parameter_.type_ = primitive->value_type();

  param->kernel_h_ = value->kernel_size()->Get(0);
  param->kernel_w_ = value->kernel_size()->Get(1);
  param->stride_h_ = value->stride()->Get((value->stride()->size()) - kInputIndexTwo);
  param->stride_w_ = value->stride()->Get((value->stride()->size()) - kInputIndexOne);
  param->dilation_h_ = value->dilation()->Get(0);
  param->dilation_w_ = value->dilation()->Get(1);
  param->pad_u_ = value->pad_list()->Get(0);
  param->pad_d_ = value->pad_list()->Get(1);
  param->pad_l_ = value->pad_list()->Get(kInputIndexTwo);
  param->pad_r_ = value->pad_list()->Get(kInputIndexThree);
  param->group_ = value->group();
  param->act_type_ = ActType_No;
  switch (value->activation_type()) {
    case schema::ActivationType_RELU:
      param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      param->act_type_ = ActType_Relu6;
      break;
    default:
      break;
  }
  switch (value->pad_mode()) {
    case schema::PadMode_SAME:
      param->pad_mode_ = Pad_same;
      break;
    case schema::PadMode_VALID:
      param->pad_mode_ = Pad_valid;
      break;
    default:
      param->pad_mode_ = Pad_pad;
      break;
  }

  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateConvolutionGradInputParameter(const void *prim) {
  ConvParameter *param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc Param for conv grad filter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ConvParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_Conv2DBackpropInputFusion();
  MS_ASSERT(value != nullptr);
  param->op_parameter_.type_ = primitive->value_type();

  param->kernel_h_ = value->kernel_size()->Get(0);
  param->kernel_w_ = value->kernel_size()->Get(1);
  param->stride_h_ = value->stride()->Get((value->stride()->size()) - kInputIndexTwo);
  param->stride_w_ = value->stride()->Get((value->stride()->size()) - kInputIndexOne);
  param->dilation_h_ = value->dilation()->Get(0);
  param->dilation_w_ = value->dilation()->Get(1);
  param->pad_u_ = value->pad_list()->Get(0);
  param->pad_d_ = value->pad_list()->Get(1);
  param->pad_l_ = value->pad_list()->Get(kInputIndexTwo);
  param->pad_r_ = value->pad_list()->Get(kInputIndexThree);
  param->group_ = value->group();
  param->act_type_ = ActType_No;
  switch (value->activation_type()) {
    case schema::ActivationType_RELU:
      param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      param->act_type_ = ActType_Relu6;
      break;
    default:
      break;
  }
  switch (value->pad_mode()) {
    case schema::PadMode_SAME:
      param->pad_mode_ = Pad_same;
      break;
    case schema::PadMode_VALID:
      param->pad_mode_ = Pad_valid;
      break;
    default:
      param->pad_mode_ = Pad_pad;
      break;
  }

  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulatePowerGradParameter(const void *prim) {
  PowerParameter *power_param = reinterpret_cast<PowerParameter *>(malloc(sizeof(PowerParameter)));
  if (power_param == nullptr) {
    MS_LOG(ERROR) << "malloc PowerParameter failed.";
    return nullptr;
  }
  memset(power_param, 0, sizeof(PowerParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_PowerGrad();
  MS_ASSERT(value != nullptr);
  power_param->op_parameter_.type_ = primitive->value_type();
  power_param->power_ = value->power();
  power_param->scale_ = value->scale();
  power_param->shift_ = value->shift();
  return reinterpret_cast<OpParameter *>(power_param);
}

OpParameter *PopulateBiasGradParameter(const void *prim) {
  ArithmeticParameter *arithmetic_param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(arithmetic_param, 0, sizeof(ArithmeticParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  arithmetic_param->op_parameter_.type_ = primitive->value_type();
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

OpParameter *PopulateBNGradParameter(const void *prim) {
  BNGradParameter *bnGrad_param = reinterpret_cast<BNGradParameter *>(malloc(sizeof(BNGradParameter)));
  if (bnGrad_param == nullptr) {
    MS_LOG(ERROR) << "malloc BNGradParameter failed.";
    return nullptr;
  }
  memset(bnGrad_param, 0, sizeof(BNGradParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_BatchNormGrad();
  MS_ASSERT(value != nullptr);
  bnGrad_param->op_parameter_.type_ = primitive->value_type();
  bnGrad_param->epsilon_ = value->epsilon();
  bnGrad_param->is_training_ = value->is_training();
  return reinterpret_cast<OpParameter *>(bnGrad_param);
}

OpParameter *PopulateDropoutParameter(const void *prim) {
  DropoutParameter *dropout_parameter = reinterpret_cast<DropoutParameter *>(malloc(sizeof(DropoutParameter)));
  if (dropout_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc Dropout Parameter failed.";
    return nullptr;
  }
  memset(dropout_parameter, 0, sizeof(DropoutParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_Dropout();
  MS_ASSERT(value != nullptr);
  dropout_parameter->op_parameter_.type_ = primitive->value_type();
  dropout_parameter->ratio_ = value->keep_prob();
  if (dropout_parameter->ratio_ < 0.f || dropout_parameter->ratio_ > 1.f) {
    MS_LOG(ERROR) << "Dropout ratio must be between 0 to 1, got " << dropout_parameter->ratio_;
    free(dropout_parameter);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(dropout_parameter);
}

OpParameter *PopulateDropoutGradParameter(const void *prim) {
  DropoutParameter *dropoutgrad_parameter = reinterpret_cast<DropoutParameter *>(malloc(sizeof(DropoutParameter)));
  if (dropoutgrad_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc Dropout Grad Parameter failed.";
    return nullptr;
  }
  memset(dropoutgrad_parameter, 0, sizeof(DropoutParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_DropoutGrad();
  MS_ASSERT(value != nullptr);
  dropoutgrad_parameter->op_parameter_.type_ = primitive->value_type();
  dropoutgrad_parameter->ratio_ = value->keep_prob();
  if (dropoutgrad_parameter->ratio_ < 0.f || dropoutgrad_parameter->ratio_ > 1.f) {
    MS_LOG(ERROR) << "Dropout Grad ratio must be between 0 to 1, got " << dropoutgrad_parameter->ratio_;
    free(dropoutgrad_parameter);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(dropoutgrad_parameter);
}

OpParameter *PopulateArithmeticGradParameter(const void *prim) {
  ArithmeticParameter *arithmetic_param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(arithmetic_param, 0, sizeof(ArithmeticParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  arithmetic_param->op_parameter_.type_ = primitive->value_type();
  return reinterpret_cast<OpParameter *>(arithmetic_param);
}

OpParameter *PopulateResizeGradParameter(const void *prim) {
  ResizeGradParameter *resize_grad_param = reinterpret_cast<ResizeGradParameter *>(malloc(sizeof(ResizeGradParameter)));
  if (resize_grad_param == nullptr) {
    MS_LOG(ERROR) << "malloc resize grad parameter failed.";
    return nullptr;
  }
  memset(resize_grad_param, 0, sizeof(ResizeGradParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  resize_grad_param->op_parameter_.type_ = primitive->value_type();
  auto param = primitive->value_as_ResizeGrad();
  MS_ASSERT(param != nullptr);
  resize_grad_param->method = static_cast<int>(param->method());
  resize_grad_param->align_corners_ = param->align_corners();

  return reinterpret_cast<OpParameter *>(resize_grad_param);
}

OpParameter *PopulateStridedSliceGradParameter(const void *prim) {
  StridedSliceParameter *strided_slice_param =
    reinterpret_cast<StridedSliceParameter *>(malloc(sizeof(StridedSliceParameter)));
  if (strided_slice_param == nullptr) {
    MS_LOG(ERROR) << "malloc StridedSliceParameter failed.";
    return nullptr;
  }
  memset(strided_slice_param, 0, sizeof(StridedSliceParameter));

  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_StridedSliceGrad();
  MS_ASSERT(value != nullptr);
  strided_slice_param->op_parameter_.type_ = primitive->value_type();

  strided_slice_param->begins_mask_ = value->begin_mask();
  strided_slice_param->ends_mask_ = value->end_mask();
  strided_slice_param->ellipsisMask_ = value->ellipsis_mask();
  strided_slice_param->newAxisMask_ = value->new_axis_mask();
  strided_slice_param->shrinkAxisMask_ = value->shrink_axis_mask();
  return reinterpret_cast<OpParameter *>(strided_slice_param);
}

OpParameter *PopulateLstmGradParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_LSTMGrad();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr.";
    return nullptr;
  }

  auto *param = reinterpret_cast<LstmGradParameter *>(malloc(sizeof(LstmGradParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc LstmGradParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(LstmGradParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->bidirectional_ = value->bidirectional();
  param->zoneout_cell_ = value->zoneout_cell();
  param->zoneout_hidden_ = value->zoneout_hidden();
  param->input_size_ = value->input_size();
  param->has_bias_ = static_cast<int>(value->has_bias());
  param->hidden_size_ = value->hidden_size();

  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateLstmGradDataParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_LSTMGradData();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr.";
    return nullptr;
  }

  auto *param = reinterpret_cast<LstmGradParameter *>(malloc(sizeof(LstmGradParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc LstmGradParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(LstmGradParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->bidirectional_ = value->bidirectional();
  param->zoneout_cell_ = value->zoneout_cell();
  param->zoneout_hidden_ = value->zoneout_hidden();
  param->input_size_ = value->input_size();
  param->has_bias_ = static_cast<int>(value->has_bias());
  param->hidden_size_ = value->hidden_size();
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateLstmGradWeightParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_LSTMGradWeight();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr.";
    return nullptr;
  }

  auto *param = reinterpret_cast<LstmGradParameter *>(malloc(sizeof(LstmGradParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc LstmGradParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(LstmGradParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->input_size_ = value->input_size();
  param->hidden_size_ = value->hidden_size();
  param->bidirectional_ = value->bidirectional();
  param->zoneout_cell_ = value->zoneout_cell();
  param->zoneout_hidden_ = value->zoneout_hidden();
  param->has_bias_ = static_cast<int>(value->has_bias());
  return reinterpret_cast<OpParameter *>(param);
}

void PopulateTrainParameters() {
  Registry ApplyMomentumParameterRegistry(schema::PrimitiveType_ApplyMomentum, PopulateApplyMomentumParameter,
                                          lite::SCHEMA_CUR);
  Registry BiasGradParameterRegistry(schema::PrimitiveType_BiasAddGrad, PopulateBiasGradParameter, lite::SCHEMA_CUR);
  Registry SoftmaxCrossEntropyParameterRegistry(schema::PrimitiveType_SoftmaxCrossEntropyWithLogits,
                                                PopulateSoftmaxCrossEntropyParameter, lite::SCHEMA_CUR);
  Registry SparseSoftmaxCrossEntropyParameterRegistry(schema::PrimitiveType_SparseSoftmaxCrossEntropyWithLogits,
                                                      PopulateSparseSoftmaxCrossEntropyWithLogitsParameter,
                                                      lite::SCHEMA_CUR);
  Registry ActivationParameterRegistry(schema::PrimitiveType_ActivationGrad, PopulateActivationGradParameter,
                                       lite::SCHEMA_CUR);
  Registry DependParameterRegistry(schema::PrimitiveType_Depend, lite::DefaultPopulateParameter, lite::SCHEMA_CUR);
  Registry Conv2DGradFilterParameterRegistry(schema::PrimitiveType_Conv2DBackpropFilterFusion,
                                             PopulateConvolutionGradFilterParameter, lite::SCHEMA_CUR);
  Registry Conv2DGradInputParameterRegistry(schema::PrimitiveType_Conv2DBackpropInputFusion,
                                            PopulateConvolutionGradInputParameter, lite::SCHEMA_CUR);
  Registry avgPoolParameterRegistry(schema::PrimitiveType_AvgPoolGrad, PopulateAvgPoolGradParameter, lite::SCHEMA_CUR);
  Registry maxPoolParameterRegistry(schema::PrimitiveType_MaxPoolGrad, PopulateMaxPoolGradParameter, lite::SCHEMA_CUR);
  Registry PowerGradParameterRegistry(schema::PrimitiveType_PowerGrad, PopulatePowerGradParameter, lite::SCHEMA_CUR);
  Registry SgdParameterRegistry(schema::PrimitiveType_SGD, PopulateSgdParameter, lite::SCHEMA_CUR);
  Registry BNGradParameterRegistry(schema::PrimitiveType_BatchNormGrad, PopulateBNGradParameter, lite::SCHEMA_CUR);
  Registry AdamParameterRegistry(schema::PrimitiveType_Adam, PopulateAdamParameter, lite::SCHEMA_CUR);
  Registry AdamWeightDecayParameterRegistry(schema::PrimitiveType_AdamWeightDecay, lite::DefaultPopulateParameter,
                                            lite::SCHEMA_CUR);
  Registry AssignParameterRegistry(schema::PrimitiveType_Assign, lite::DefaultPopulateParameter, lite::SCHEMA_CUR);
  Registry AssignAddParameterRegistry(schema::PrimitiveType_AssignAdd, lite::DefaultPopulateParameter,
                                      lite::SCHEMA_CUR);
  Registry BinaryCrossEntropyParameterRegistry(schema::PrimitiveType_BinaryCrossEntropy, PopulateBCEParameter,
                                               lite::SCHEMA_CUR);
  Registry BinaryCrossEntropyGradParameterRegistry(schema::PrimitiveType_BinaryCrossEntropyGrad,
                                                   PopulateBCEGradParameter, lite::SCHEMA_CUR);
  Registry OnesLikeParameterRegistry(schema::PrimitiveType_OnesLike, lite::DefaultPopulateParameter, lite::SCHEMA_CUR);
  Registry UnsortedSegmentSumParameterRegistry(schema::PrimitiveType_UnsortedSegmentSum, lite::DefaultPopulateParameter,
                                               lite::SCHEMA_CUR);
  Registry DropoutParameterRegistry(schema::PrimitiveType_Dropout, PopulateDropoutParameter, lite::SCHEMA_CUR);
  Registry DropGradParameterRegistry(schema::PrimitiveType_DropoutGrad, PopulateDropoutGradParameter, lite::SCHEMA_CUR);
  Registry MaximumGradParameterRegistry(schema::PrimitiveType_MaximumGrad, PopulateArithmeticGradParameter,
                                        lite::SCHEMA_CUR);
  Registry MinimumGradParameterRegistry(schema::PrimitiveType_MinimumGrad, PopulateArithmeticGradParameter,
                                        lite::SCHEMA_CUR);
  Registry SmoothL1LossRegistry(schema::PrimitiveType_SmoothL1Loss, PopulateSmoothL1LossParameter, lite::SCHEMA_CUR);
  Registry SmoothL1LossGradRegistry(schema::PrimitiveType_SmoothL1LossGrad, PopulateSmoothL1LossGradParameter,
                                    lite::SCHEMA_CUR);
  Registry SigmoidCrossEntropyWithLogitsRegistry(schema::PrimitiveType_SigmoidCrossEntropyWithLogits,
                                                 lite::DefaultPopulateParameter, lite::SCHEMA_CUR);
  Registry SigmoidCrossEntropyWithLogitsGradRegistry(schema::PrimitiveType_SigmoidCrossEntropyWithLogitsGrad,
                                                     lite::DefaultPopulateParameter, lite::SCHEMA_CUR);
  Registry FlattenGradParameterRegistry(schema::PrimitiveType_FlattenGrad, lite::DefaultPopulateParameter,
                                        lite::SCHEMA_CUR);
  Registry StridedSliceGradParameterRegistry(schema::PrimitiveType_StridedSliceGrad, PopulateStridedSliceGradParameter,
                                             lite::SCHEMA_CUR);
  Registry SqrtGradParameterRegistry(schema::PrimitiveType_SqrtGrad, lite::DefaultPopulateParameter, lite::SCHEMA_CUR);
  Registry RsqrtGradParameterRegistry(schema::PrimitiveType_RsqrtGrad, lite::DefaultPopulateParameter,
                                      lite::SCHEMA_CUR);
  Registry ResizeGradParameterRegistry(schema::PrimitiveType_ResizeGrad, PopulateResizeGradParameter, lite::SCHEMA_CUR);
  Registry AbsGradParameterRegistry(schema::PrimitiveType_AbsGrad, lite::DefaultPopulateParameter, lite::SCHEMA_CUR);
  Registry LSTMGradParameterRegistry(schema::PrimitiveType_LSTMGrad, PopulateLstmGradParameter, lite::SCHEMA_CUR);
  Registry LSTMGradDataParameterRegistry(schema::PrimitiveType_LSTMGradData, PopulateLstmGradDataParameter,
                                         lite::SCHEMA_CUR);
  Registry LSTMGradWeightParameterRegistry(schema::PrimitiveType_LSTMGradWeight, PopulateLstmGradWeightParameter,
                                           lite::SCHEMA_CUR);
}
}  // namespace kernel
}  // namespace mindspore
