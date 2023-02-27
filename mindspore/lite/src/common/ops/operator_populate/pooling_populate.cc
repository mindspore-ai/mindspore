/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/pooling_parameter.h"
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fusion/max_pool_fusion.h"
using mindspore::ops::kNameAvgPoolFusion;
using mindspore::ops::kNameMaxPoolFusion;
using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;

namespace mindspore {
namespace lite {
namespace {
int CheckOpPoolingParam(const PoolingParameter *param) {
  const int max_pooling_pad = 50;
  if (param->pad_u_ > max_pooling_pad || param->pad_d_ > max_pooling_pad || param->pad_l_ > max_pooling_pad ||
      param->pad_r_ > max_pooling_pad) {
    return RET_ERROR;
  }
  return RET_OK;
}

void UpdateOpRoundMode(RoundMode round_mode, PoolingParameter *param) {
  switch (round_mode) {
    case FLOOR:
      param->round_mode_ = RoundMode_Floor;
      break;
    case CEIL:
      param->round_mode_ = RoundMode_Ceil;
      break;
    default:
      param->round_mode_ = RoundMode_No;
      break;
  }
}

void UpdateOpActivationType(ActivationType type, PoolingParameter *param) {
  if (type == RELU) {
    param->act_type_ = ActType_Relu;
  } else if (type == RELU6) {
    param->act_type_ = ActType_Relu6;
  } else {
    param->act_type_ = ActType_No;
  }
}

void UpdateOpPadMode(PadMode pad_mode, PoolingParameter *param) {
  switch (pad_mode) {
    case SAME:
      param->pad_mode_ = Pad_same;
      break;
    case VALID:
      param->pad_mode_ = Pad_valid;
      break;
    default:
      param->pad_mode_ = Pad_pad;
      break;
  }
}
}  // namespace
OpParameter *PopulateAvgPoolOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<PoolingParameter *>(PopulateOpParameter<PoolingParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new PoolingParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::AvgPoolFusion *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not AvgPoolFusion.";
    return nullptr;
  }

  param->pool_mode_ = PoolMode_AvgPool;
  param->global_ = op->get_global();
  auto strides = op->get_strides();
  if (strides.size() < DIMENSION_2D) {
    MS_LOG(ERROR) << "strides is invalid!";
    free(param);
    return nullptr;
  }
  param->stride_w_ = static_cast<int>(strides[1]);
  param->stride_h_ = static_cast<int>(strides[0]);
  auto pad = op->get_pad();
  if (pad.size() >= DIMENSION_4D) {
    param->pad_u_ = static_cast<int>(*(pad.begin()));
    param->pad_d_ = static_cast<int>(*(pad.begin() + 1));
    param->pad_l_ = static_cast<int>(*(pad.begin() + C2NUM));
    param->pad_r_ = static_cast<int>(*(pad.begin() + C3NUM));
  }
  if (!param->global_) {
    auto kernel_size = op->get_kernel_size();
    if (kernel_size.size() < DIMENSION_2D) {
      MS_LOG(ERROR) << "kernel_size is invalid";
      free(param);
      return nullptr;
    }
    param->window_w_ = static_cast<int>(*(kernel_size.begin() + 1));
    param->window_h_ = static_cast<int>(*(kernel_size.begin()));
  }

  UpdateOpRoundMode(op->get_round_mode(), param);
  UpdateOpActivationType(op->get_activation_type(), param);
  UpdateOpPadMode(op->get_pad_mode(), param);

  if (CheckOpPoolingParam(param) != RET_OK) {
    MS_LOG(ERROR) << "param is invalid!";
    free(param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateMaxPoolOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<PoolingParameter *>(PopulateOpParameter<PoolingParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new PoolingParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::MaxPoolFusion *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not MaxPoolFusion.";
    return nullptr;
  }

  param->pool_mode_ = PoolMode_MaxPool;
  param->global_ = op->get_global();
  if (!param->global_) {
    auto kernel_size = op->get_kernel_size();
    auto strides = op->get_strides();
    if (kernel_size.size() < DIMENSION_2D || strides.size() < DIMENSION_2D) {
      MS_LOG(ERROR) << "kernel_size or strides is invalid";
      free(param);
      return nullptr;
    }
    param->window_w_ = static_cast<int>(*(kernel_size.begin() + 1));
    param->window_h_ = static_cast<int>(*(kernel_size.begin()));
    param->stride_w_ = static_cast<int>(*(strides.begin() + 1));
    param->stride_h_ = static_cast<int>(*(strides.begin()));
    auto pad = op->get_pad();
    if (pad.size() >= DIMENSION_4D) {
      param->pad_u_ = static_cast<int>(*(pad.begin()));
      param->pad_d_ = static_cast<int>(*(pad.begin() + 1));
      param->pad_l_ = static_cast<int>(*(pad.begin() + C2NUM));
      param->pad_r_ = static_cast<int>(*(pad.begin() + C3NUM));
    }
  }

  UpdateOpRoundMode(op->get_round_mode(), param);
  UpdateOpActivationType(op->get_activation_type(), param);
  UpdateOpPadMode(op->get_pad_mode(), param);

  if (CheckOpPoolingParam(param) != RET_OK) {
    MS_LOG(ERROR) << "param is invalid!";
    free(param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameAvgPoolFusion, PrimitiveType_AvgPoolFusion, PopulateAvgPoolOpParameter)
REG_OPERATOR_POPULATE(kNameMaxPoolFusion, PrimitiveType_MaxPoolFusion, PopulateMaxPoolOpParameter)
}  // namespace lite
}  // namespace mindspore
