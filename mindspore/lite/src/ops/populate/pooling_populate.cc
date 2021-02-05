/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/pooling.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/pooling_parameter.h"

namespace mindspore {
namespace lite {

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
  pooling_param->avg_mode_ = pooling_primitive->GetAvgMode();
  auto pad_mode = pooling_primitive->GetPadMode();
  switch (pad_mode) {
    case schema::PadMode_SAME_UPPER:
      pooling_param->pad_mode_ = Pad_Same;
      break;
    case schema::PadMode_VALID:
      pooling_param->pad_mode_ = Pad_Valid;
      break;
    default:
      pooling_param->pad_mode_ = Pad_No;
      break;
  }

  auto is_global = pooling_primitive->GetGlobal();
  pooling_param->global_ = is_global;
  auto pool_mode = pooling_primitive->GetPoolingMode();
  switch (pool_mode) {
    case schema::PoolMode_MAX_POOLING:
      pooling_param->pool_mode_ = PoolMode_MaxPool;
      break;
    case schema::PoolMode_MEAN_POOLING:
      pooling_param->pool_mode_ = PoolMode_AvgPool;
      break;
    default:
      pooling_param->pool_mode_ = PoolMode_No;
      break;
  }

  auto round_mode = pooling_primitive->GetRoundMode();
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

  if (pooling_primitive->GetActivationType() == schema::ActivationType_RELU) {
    pooling_param->act_type_ = ActType_Relu;
  } else if (pooling_primitive->GetActivationType() == schema::ActivationType_RELU6) {
    pooling_param->act_type_ = ActType_Relu6;
  } else {
    pooling_param->act_type_ = ActType_No;
  }
  return reinterpret_cast<OpParameter *>(pooling_param);
}

Registry PoolingParameterRegistry(schema::PrimitiveType_Pooling, PopulatePoolingParameter);

}  // namespace lite
}  // namespace mindspore
