/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/pooling_parameter.h"
using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;

namespace mindspore {
namespace lite {
namespace {
void UpdateRoundMode(enum schema::RoundMode round_mode, PoolingParameter *param) {
  switch (round_mode) {
    case schema::RoundMode_FLOOR:
      param->round_mode_ = RoundMode_Floor;
      break;
    case schema::RoundMode_CEIL:
      param->round_mode_ = RoundMode_Ceil;
      break;
    default:
      param->round_mode_ = RoundMode_No;
      break;
  }
}

void UpdateActivationType(enum schema::ActivationType type, PoolingParameter *param) {
  if (type == schema::ActivationType_RELU) {
    param->act_type_ = ActType_Relu;
  } else if (type == schema::ActivationType_RELU6) {
    param->act_type_ = ActType_Relu6;
  } else {
    param->act_type_ = ActType_No;
  }
}

void UpdatePadMode(enum schema::PadMode pad_mode, PoolingParameter *param) {
  switch (pad_mode) {
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
}
}  // namespace
OpParameter *PopulateAvgPoolParameter(const void *primitive) {
  MS_CHECK_TRUE_RET(primitive != nullptr, nullptr);
  auto pooling_prim = static_cast<const schema::Primitive *>(primitive);
  auto value = pooling_prim->value_as_AvgPoolFusion();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc PoolingParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(PoolingParameter));

  param->op_parameter_.type_ = pooling_prim->value_type();
  param->pool_mode_ = PoolMode_AvgPool;
  param->global_ = value->global();
  auto strides = value->strides();
  if (strides == nullptr || strides->size() < kMinShapeSizeTwo) {
    MS_LOG(ERROR) << "strides is invalid!";
    free(param);
    return nullptr;
  }
  param->stride_w_ = static_cast<int>(*(strides->begin() + 1));
  param->stride_h_ = static_cast<int>(*(strides->begin()));
  auto pad = value->pad();
  if (pad != nullptr && pad->size() >= kMinShapeSizeFour) {
    param->pad_u_ = static_cast<int>(*(pad->begin()));
    param->pad_d_ = static_cast<int>(*(pad->begin() + 1));
    param->pad_l_ = static_cast<int>(*(pad->begin() + kOffsetTwo));
    param->pad_r_ = static_cast<int>(*(pad->begin() + kOffsetThree));
  }
  if (!param->global_) {
    auto kernel_size = value->kernel_size();
    if (kernel_size == nullptr || kernel_size->size() < kMinShapeSizeTwo) {
      MS_LOG(ERROR) << "kernel_size is invalid";
      free(param);
      return nullptr;
    }
    param->window_w_ = static_cast<int>(*(kernel_size->begin() + 1));
    param->window_h_ = static_cast<int>(*(kernel_size->begin()));
  }

  UpdateRoundMode(value->round_mode(), param);
  UpdateActivationType(value->activation_type(), param);
  UpdatePadMode(value->pad_mode(), param);
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *PopulateMaxPoolParameter(const void *primitive) {
  auto pooling_prim = static_cast<const schema::Primitive *>(primitive);
  MS_ASSERT(pooling_prim != nullptr);
  auto value = pooling_prim->value_as_MaxPoolFusion();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc PoolingParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(PoolingParameter));

  param->op_parameter_.type_ = pooling_prim->value_type();
  param->pool_mode_ = PoolMode_MaxPool;
  param->global_ = value->global();
  if (!param->global_) {
    auto kernel_size = value->kernel_size();
    auto strides = value->strides();
    if (kernel_size == nullptr || strides == nullptr || kernel_size->size() < kMinShapeSizeTwo ||
        strides->size() < kMinShapeSizeTwo) {
      MS_LOG(ERROR) << "kernel_size or strides is invalid";
      free(param);
      return nullptr;
    }
    param->window_w_ = static_cast<int>(*(kernel_size->begin() + 1));
    param->window_h_ = static_cast<int>(*(kernel_size->begin()));
    param->stride_w_ = static_cast<int>(*(strides->begin() + 1));
    param->stride_h_ = static_cast<int>(*(strides->begin()));
    auto pad = value->pad();
    if (pad != nullptr && pad->size() >= kMinShapeSizeFour) {
      param->pad_u_ = static_cast<int>(*(pad->begin()));
      param->pad_d_ = static_cast<int>(*(pad->begin() + 1));
      param->pad_l_ = static_cast<int>(*(pad->begin() + kOffsetTwo));
      param->pad_r_ = static_cast<int>(*(pad->begin() + kOffsetThree));
    }
  }

  UpdateRoundMode(value->round_mode(), param);
  UpdateActivationType(value->activation_type(), param);
  UpdatePadMode(value->pad_mode(), param);
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_AvgPoolFusion, PopulateAvgPoolParameter, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_MaxPoolFusion, PopulateMaxPoolParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
