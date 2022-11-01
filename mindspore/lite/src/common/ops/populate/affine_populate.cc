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
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/op_base.h"
#include "nnacl/affine_parameter.h"

using mindspore::schema::PrimitiveType_Affine;

namespace mindspore {
namespace lite {
void DestroyAffineParas(OpParameter *parameter) {
  MS_CHECK_PTR_IF_NULL(parameter);
  MS_LOG(INFO) << "Destroy affine paras";
  auto param = reinterpret_cast<AffineParameter *>(parameter);
  if (param->matmul_parameter_ != nullptr) {
    free(param->matmul_parameter_);
    param->matmul_parameter_ = nullptr;
  }
  if (param->context_ != nullptr) {
    free(param->context_);
    param->context_ = nullptr;
  }
}

static void ReleaseParam(AffineParameter *affine, MatMulParameter *matmul) {
  if (affine != nullptr) {
    free(affine);
  }
  if (matmul != nullptr) {
    free(matmul);
  }
}

OpParameter *PopulateAffineParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_Affine();
  if (value == nullptr) {
    MS_LOG(ERROR) << "cast affine_primitive to value failed";
    return nullptr;
  }
  auto *affine_param = reinterpret_cast<AffineParameter *>(malloc(sizeof(AffineParameter)));
  if (affine_param == nullptr) {
    MS_LOG(ERROR) << "malloc Affine Parameter failed.";
    return nullptr;
  }
  memset(affine_param, 0, sizeof(AffineParameter));
  affine_param->op_parameter_.destroy_func_ = DestroyAffineParas;
  auto *matmul_param = reinterpret_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "malloc MatMulParameter failed.";
    ReleaseParam(affine_param, nullptr);
    return nullptr;
  }
  memset(matmul_param, 0, sizeof(MatMulParameter));
  matmul_param->op_parameter_.type_ = primitive->value_type();
  matmul_param->b_transpose_ = value->transpose_b();
  matmul_param->a_transpose_ = value->transpose_a();
  matmul_param->has_bias_ = false;
  matmul_param->act_type_ = ActType_No;

  affine_param->matmul_parameter_ = matmul_param;
  affine_param->op_parameter_.type_ = primitive->value_type();
  affine_param->activation_type_ = static_cast<int>(value->activation_type());
  auto context_attr = value->context();
  if (context_attr == nullptr) {
    MS_LOG(ERROR) << "context is nullptr";
    ReleaseParam(affine_param, matmul_param);
    return nullptr;
  }
  std::vector<int> context(context_attr->begin(), context_attr->end());
  affine_param->context_size_ = static_cast<int>(context.size());

  // malloc && memset for context
  affine_param->context_ = reinterpret_cast<int *>(malloc(context.size() * sizeof(int)));
  if (affine_param->context_ == nullptr) {
    MS_LOG(ERROR) << "malloc param context_ for affine layer failed!";
    ReleaseParam(affine_param, matmul_param);
    return nullptr;
  }
  (void)memset(affine_param->context_, 0, context.size() * sizeof(int));
  for (size_t i = 0; i < context.size(); ++i) {
    affine_param->context_[i] = context.at(i);
  }
  affine_param->output_dim_ = value->output_dim();
  return reinterpret_cast<OpParameter *>(affine_param);
}

REG_POPULATE(PrimitiveType_Affine, PopulateAffineParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
