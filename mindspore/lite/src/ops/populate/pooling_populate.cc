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
OpParameter *PopulateAvgPoolParameter(const void *primitive) {
  auto *pooling_param = reinterpret_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  if (pooling_param == nullptr) {
    MS_LOG(ERROR) << "malloc PoolingParameter failed.";
    return nullptr;
  }
  memset(pooling_param, 0, sizeof(PoolingParameter));
  auto pooling_prim = static_cast<const schema::Primitive *>(primitive);
  MS_ASSERT(pooling_prim != nullptr);
  pooling_param->op_parameter_.type_ = pooling_prim->value_type();
  auto pooling_primitive = pooling_prim->value_as_AvgPoolFusion();
  if (pooling_primitive == nullptr) {
    MS_LOG(ERROR) << "pooling_primitive is nullptr";
    return nullptr;
  }
  pooling_param->pool_mode_ = PoolMode_AvgPool;
  pooling_param->global_ = pooling_primitive->global();
  auto strides = pooling_primitive->strides();
  if (strides == nullptr) {
    MS_LOG(ERROR) << "strides is nullptr";
    return nullptr;
  }
  pooling_param->stride_w_ = static_cast<int>(*(strides->begin() + 1));
  pooling_param->stride_h_ = static_cast<int>(*(strides->begin()));
  auto pad = pooling_primitive->pad();
  if (pad != nullptr) {
    pooling_param->pad_u_ = static_cast<int>(*(pad->begin()));
    pooling_param->pad_d_ = static_cast<int>(*(pad->begin() + 1));
    pooling_param->pad_l_ = static_cast<int>(*(pad->begin() + 2));
    pooling_param->pad_r_ = static_cast<int>(*(pad->begin() + 3));
  }
  if (!pooling_param->global_) {
    auto kernel_size = pooling_primitive->kernel_size();
    if (kernel_size == nullptr) {
      MS_LOG(ERROR) << "kernel_size is nullptr";
      return nullptr;
    }
    pooling_param->window_w_ = static_cast<int>(*(kernel_size->begin() + 1));
    pooling_param->window_h_ = static_cast<int>(*(kernel_size->begin()));
  }

  auto round_mode = pooling_primitive->round_mode();
  switch (round_mode) {
    case schema::RoundMode_FLOOR:
      pooling_param->round_mode_ = RoundMode_Floor;
      break;
    case schema::RoundMode_CEIL:
      pooling_param->round_mode_ = RoundMode_Ceil;
      break;
    default:
      pooling_param->round_mode_ = RoundMode_No;
      break;
  }

  if (pooling_primitive->activation_type() == schema::ActivationType_RELU) {
    pooling_param->act_type_ = ActType_Relu;
  } else if (pooling_primitive->activation_type() == schema::ActivationType_RELU6) {
    pooling_param->act_type_ = ActType_Relu6;
  } else {
    pooling_param->act_type_ = ActType_No;
  }

  switch (pooling_primitive->pad_mode()) {
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

OpParameter *PopulateMaxPoolParameter(const void *primitive) {
  auto *pooling_param = reinterpret_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  if (pooling_param == nullptr) {
    MS_LOG(ERROR) << "malloc PoolingParameter failed.";
    return nullptr;
  }
  memset(pooling_param, 0, sizeof(PoolingParameter));
  auto pooling_prim = static_cast<const schema::Primitive *>(primitive);
  MS_ASSERT(pooling_prim != nullptr);
  pooling_param->op_parameter_.type_ = pooling_prim->value_type();
  auto max_pool_prim = pooling_prim->value_as_MaxPoolFusion();
  if (max_pool_prim == nullptr) {
    MS_LOG(ERROR) << "max_pool_prim is nullptr";
    return nullptr;
  }
  pooling_param->pool_mode_ = PoolMode_MaxPool;
  pooling_param->global_ = max_pool_prim->global();
  if (!pooling_param->global_) {
    auto kernel_size = max_pool_prim->kernel_size();
    auto strides = max_pool_prim->strides();
    if (kernel_size == nullptr || strides == nullptr) {
      MS_LOG(ERROR) << "kernel_size or strides is nullptr";
      return nullptr;
    }
    pooling_param->window_w_ = static_cast<int>(*(kernel_size->begin() + 1));
    pooling_param->window_h_ = static_cast<int>(*(kernel_size->begin()));
    pooling_param->stride_w_ = static_cast<int>(*(strides->begin() + 1));
    pooling_param->stride_h_ = static_cast<int>(*(strides->begin()));
    auto pad = max_pool_prim->pad();
    if (pad != nullptr) {
      pooling_param->pad_u_ = static_cast<int>(*(pad->begin()));
      pooling_param->pad_d_ = static_cast<int>(*(pad->begin() + 1));
      pooling_param->pad_l_ = static_cast<int>(*(pad->begin() + 2));
      pooling_param->pad_r_ = static_cast<int>(*(pad->begin() + 3));
    }
  }

  auto round_mode = max_pool_prim->round_mode();
  switch (round_mode) {
    case schema::RoundMode_FLOOR:
      pooling_param->round_mode_ = RoundMode_Floor;
      break;
    case schema::RoundMode_CEIL:
      pooling_param->round_mode_ = RoundMode_Ceil;
      break;
    default:
      pooling_param->round_mode_ = RoundMode_No;
      break;
  }

  if (max_pool_prim->activation_type() == schema::ActivationType_RELU) {
    pooling_param->act_type_ = ActType_Relu;
  } else if (max_pool_prim->activation_type() == schema::ActivationType_RELU6) {
    pooling_param->act_type_ = ActType_Relu6;
  } else {
    pooling_param->act_type_ = ActType_No;
  }

  switch (max_pool_prim->pad_mode()) {
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
}  // namespace

REG_POPULATE(PrimitiveType_AvgPoolFusion, PopulateAvgPoolParameter, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_MaxPoolFusion, PopulateMaxPoolParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
