/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/pooling_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulatePoolingParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto pooling_prim = primitive->value_as_Pooling();

  PoolingParameter *pooling_param = reinterpret_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  if (pooling_param == nullptr) {
    MS_LOG(ERROR) << "malloc PoolingParameter failed.";
    return nullptr;
  }
  memset(pooling_param, 0, sizeof(PoolingParameter));
  pooling_param->global_ = pooling_prim->global();
  pooling_param->window_w_ = pooling_prim->windowW();
  pooling_param->window_h_ = pooling_prim->windowH();
  pooling_param->pad_u_ = pooling_prim->padUp();
  pooling_param->pad_d_ = pooling_prim->padDown();
  pooling_param->pad_l_ = pooling_prim->padLeft();
  pooling_param->pad_r_ = pooling_prim->padRight();
  pooling_param->stride_w_ = pooling_prim->strideW();
  pooling_param->stride_h_ = pooling_prim->strideH();
  pooling_param->avg_mode_ = pooling_prim->avgMode();

  auto is_global = pooling_prim->global();
  pooling_param->global_ = is_global;
  auto pool_mode = pooling_prim->poolingMode();
  switch (pool_mode) {
    case schema::v0::PoolMode_MAX_POOLING:
      pooling_param->pool_mode_ = PoolMode_MaxPool;
      pooling_param->op_parameter_.type_ = schema::PrimitiveType_MaxPoolFusion;
      break;
    case schema::v0::PoolMode_MEAN_POOLING:
      pooling_param->pool_mode_ = PoolMode_AvgPool;
      pooling_param->op_parameter_.type_ = schema::PrimitiveType_AvgPoolFusion;
      break;
    default:
      pooling_param->pool_mode_ = PoolMode_No;
      pooling_param->op_parameter_.type_ = primitive->value_type();
      break;
  }

  auto round_mode = pooling_prim->roundMode();
  switch (round_mode) {
    case schema::v0::RoundMode_FLOOR:
      pooling_param->round_mode_ = RoundMode_Floor;
      break;
    case schema::v0::RoundMode_CEIL:
      pooling_param->round_mode_ = RoundMode_Ceil;
      break;
    default:
      pooling_param->round_mode_ = RoundMode_No;
      break;
  }

  if (pooling_prim->activationType() == schema::v0::ActivationType_RELU) {
    pooling_param->act_type_ = ActType_Relu;
  } else if (pooling_prim->activationType() == schema::v0::ActivationType_RELU6) {
    pooling_param->act_type_ = ActType_Relu6;
  } else {
    pooling_param->act_type_ = ActType_No;
  }
  switch (pooling_prim->padMode()) {
    case schema::v0::PadMode_SAME_UPPER:
      pooling_param->pad_mode_ = Pad_same;
      break;
    case schema::v0::PadMode_VALID:
      pooling_param->pad_mode_ = Pad_valid;
      break;
    default:
      pooling_param->pad_mode_ = Pad_pad;
      break;
  }
  return reinterpret_cast<OpParameter *>(pooling_param);
}
}  // namespace

Registry g_poolingV0ParameterRegistry(schema::v0::PrimitiveType_Pooling, PopulatePoolingParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
