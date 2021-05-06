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
#include "nnacl/prior_box_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulatePriorBoxParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto prior_box_prim = primitive->value_as_PriorBox();
  if (prior_box_prim == nullptr) {
    MS_LOG(ERROR) << "prior_box_prim is nullptr";
    return nullptr;
  }
  auto *prior_box_param = reinterpret_cast<PriorBoxParameter *>(malloc(sizeof(PriorBoxParameter)));
  if (prior_box_param == nullptr) {
    MS_LOG(ERROR) << "malloc PriorBoxParameter failed.";
    return nullptr;
  }
  memset(prior_box_param, 0, sizeof(PriorBoxParameter));
  prior_box_param->op_parameter_.type_ = schema::PrimitiveType_PriorBox;

  auto min_sizes = prior_box_prim->min_sizes();
  if (min_sizes == nullptr) {
    MS_LOG(ERROR) << "min_sizes is nullptr";
    free(prior_box_param);
    return nullptr;
  }
  if (min_sizes->size() > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "PriorBox min_sizes size exceeds max num " << MAX_SHAPE_SIZE << ", got " << min_sizes;
    free(prior_box_param);
    return nullptr;
  }
  prior_box_param->min_sizes_size = min_sizes->size();
  memcpy(prior_box_param->min_sizes, min_sizes->data(), min_sizes->size() * sizeof(int32_t));

  auto max_sizes = prior_box_prim->max_sizes();
  if (max_sizes == nullptr) {
    MS_LOG(ERROR) << "max_sizes is nullptr";
    free(prior_box_param);
    return nullptr;
  }
  if (max_sizes->size() > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "PriorBox max_sizes size exceeds max num " << MAX_SHAPE_SIZE << ", got " << max_sizes;
    free(prior_box_param);
    return nullptr;
  }
  prior_box_param->max_sizes_size = max_sizes->size();
  memcpy(prior_box_param->max_sizes, max_sizes->data(), max_sizes->size() * sizeof(int32_t));

  auto aspect_ratios = prior_box_prim->aspect_ratios();
  if (aspect_ratios == nullptr) {
    MS_LOG(ERROR) << "aspect_ratios is nullptr";
    free(prior_box_param);
    return nullptr;
  }
  if (aspect_ratios->size() > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "PriorBox aspect_ratios size exceeds max num " << MAX_SHAPE_SIZE << ", got " << aspect_ratios;
    free(prior_box_param);
    return nullptr;
  }
  prior_box_param->aspect_ratios_size = aspect_ratios->size();
  memcpy(prior_box_param->aspect_ratios, aspect_ratios->data(), aspect_ratios->size() * sizeof(float));

  auto variances = prior_box_prim->variances();
  if (variances == nullptr) {
    MS_LOG(ERROR) << "variances is nullptr";
    free(prior_box_param);
    return nullptr;
  }
  if (variances->size() != COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "PriorBox variances size should be " << COMM_SHAPE_SIZE << ", got " << variances->size();
    free(prior_box_param);
    return nullptr;
  }
  memcpy(prior_box_param->variances, variances->data(), COMM_SHAPE_SIZE * sizeof(float));

  prior_box_param->flip = prior_box_prim->flip();
  prior_box_param->clip = prior_box_prim->clip();
  prior_box_param->offset = prior_box_prim->offset();
  prior_box_param->image_size_h = prior_box_prim->image_size_h();
  prior_box_param->image_size_w = prior_box_prim->image_size_w();
  prior_box_param->step_h = prior_box_prim->step_h();
  prior_box_param->step_w = prior_box_prim->step_w();
  return reinterpret_cast<OpParameter *>(prior_box_param);
}
}  // namespace

Registry g_priorBoxV0ParameterRegistry(schema::v0::PrimitiveType_PriorBox, PopulatePriorBoxParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
