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
#include "nnacl/prior_box_parameter.h"
#include "ops/prior_box.h"
using mindspore::ops::kNamePriorBox;
using mindspore::schema::PrimitiveType_PriorBox;

namespace mindspore {
namespace lite {
OpParameter *PopulatePriorBoxOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<PriorBoxParameter *>(PopulateOpParameter<PriorBoxParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new PriorBoxParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::PriorBox *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not PriorBox.";
    return nullptr;
  }
  auto min_sizes = op->get_min_sizes();
  if (min_sizes.size() > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "PriorBox min_sizes size exceeds max num " << MAX_SHAPE_SIZE << ", got " << min_sizes.size();
    free(param);
    return nullptr;
  }
  param->min_sizes_size = static_cast<int32_t>(min_sizes.size());
  for (int i = 0; i < param->min_sizes_size; i++) {
    param->min_sizes[i] = static_cast<int>(min_sizes[i]);
  }

  auto max_sizes = op->get_max_sizes();
  if (max_sizes.size() > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "PriorBox max_sizes size exceeds max num " << MAX_SHAPE_SIZE << ", got " << max_sizes.size();
    free(param);
    return nullptr;
  }
  param->max_sizes_size = static_cast<int32_t>(max_sizes.size());
  for (int i = 0; i < param->max_sizes_size; i++) {
    param->max_sizes[i] = static_cast<int>(max_sizes[i]);
  }

  auto aspect_ratios = op->get_aspect_ratios();
  if (aspect_ratios.size() > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "PriorBox aspect_ratios size exceeds max num " << MAX_SHAPE_SIZE << ", got "
                  << aspect_ratios.size();
    free(param);
    return nullptr;
  }
  param->aspect_ratios_size = static_cast<int32_t>(aspect_ratios.size());
  memcpy(param->aspect_ratios, aspect_ratios.data(), aspect_ratios.size() * sizeof(float));

  auto variances = op->get_variances();
  if (variances.size() != COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "PriorBox variances size should be " << COMM_SHAPE_SIZE << ", got " << variances.size();
    free(param);
    return nullptr;
  }
  memcpy(param->variances, variances.data(), COMM_SHAPE_SIZE * sizeof(float));
  param->flip = op->get_flip();
  param->clip = op->get_clip();
  param->offset = op->get_offset();
  param->image_size_h = static_cast<int>(op->get_image_size_h());
  param->image_size_w = static_cast<int>(op->get_image_size_w());
  param->step_h = op->get_step_h();
  param->step_w = op->get_step_w();
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNamePriorBox, PrimitiveType_PriorBox, PopulatePriorBoxOpParameter)
}  // namespace lite
}  // namespace mindspore
