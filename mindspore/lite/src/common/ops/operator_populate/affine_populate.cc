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
#include "nnacl/affine_parameter.h"
#include "ops/affine.h"
using mindspore::ops::kNameAffine;
using mindspore::schema::PrimitiveType_Affine;
namespace mindspore {
namespace lite {
namespace {
void DestroyAffineOpParas(OpParameter *parameter) {
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

static void ReleaseOpParam(AffineParameter *affine, MatMulParameter *matmul) {
  if (affine != nullptr) {
    free(affine);
  }
  if (matmul != nullptr) {
    free(matmul);
  }
}
}  // namespace

OpParameter *PopulateAffineOpParameter(const BaseOperatorPtr &base_operator) {
  auto affine_param = reinterpret_cast<AffineParameter *>(PopulateOpParameter<AffineParameter>(base_operator));
  if (affine_param == nullptr) {
    MS_LOG(ERROR) << "new AffineParameter failed.";
    return nullptr;
  }
  affine_param->op_parameter_.destroy_func_ = DestroyAffineOpParas;

  auto matmul_param = reinterpret_cast<MatMulParameter *>(PopulateOpParameter<MatMulParameter>(base_operator));
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "new MatMulParameter failed.";
    ReleaseOpParam(affine_param, nullptr);
    return nullptr;
  }

  auto op = dynamic_cast<ops::Affine *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to Affine failed";
    ReleaseOpParam(affine_param, matmul_param);
    return nullptr;
  }
  matmul_param->b_transpose_ = op->get_transpose_b();
  matmul_param->a_transpose_ = op->get_transpose_a();
  matmul_param->has_bias_ = false;
  matmul_param->act_type_ = ActType_No;

  affine_param->matmul_parameter_ = matmul_param;
  affine_param->activation_type_ = static_cast<int>(op->get_activation_type());

  auto context = op->get_context();
  affine_param->context_size_ = context.size();

  // malloc && memset for context
  affine_param->context_ = reinterpret_cast<int *>(malloc(context.size() * sizeof(int)));
  if (affine_param->context_ == nullptr) {
    MS_LOG(ERROR) << "malloc param context_ for affine layer failed!";
    ReleaseOpParam(affine_param, matmul_param);
    return nullptr;
  }
  (void)memset(affine_param->context_, 0, context.size() * sizeof(int));
  for (size_t i = 0; i < context.size(); ++i) {
    affine_param->context_[i] = context.at(i);
  }
  affine_param->output_dim_ = op->get_output_dim();
  return reinterpret_cast<OpParameter *>(affine_param);
}

REG_OPERATOR_POPULATE(kNameAffine, PrimitiveType_Affine, PopulateAffineOpParameter)
}  // namespace lite
}  // namespace mindspore
