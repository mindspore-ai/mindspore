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

#include "src/ops/prior_box.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "mindspore/lite/nnacl/prior_box_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulatePriorBoxParameter(const mindspore::lite::PrimitiveC *primitive) {
  PriorBoxParameter *prior_box_param = reinterpret_cast<PriorBoxParameter *>(malloc(sizeof(PriorBoxParameter)));
  if (prior_box_param == nullptr) {
    MS_LOG(ERROR) << "malloc PriorBoxParameter failed.";
    return nullptr;
  }
  memset(prior_box_param, 0, sizeof(PriorBoxParameter));
  prior_box_param->op_parameter_.type_ = primitive->Type();
  auto prior_box_attr =
    reinterpret_cast<mindspore::lite::PriorBox *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));

  if (prior_box_attr->GetMinSizes().size() > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "PriorBox min_sizes size exceeds max num " << MAX_SHAPE_SIZE << ", got "
                  << prior_box_attr->GetMinSizes();
    free(prior_box_param);
    return nullptr;
  }
  prior_box_param->min_sizes_size = prior_box_attr->GetMinSizes().size();
  if (prior_box_attr->GetMaxSizes().size() > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "PriorBox max_sizes size exceeds max num " << MAX_SHAPE_SIZE << ", got "
                  << prior_box_attr->GetMaxSizes();
    free(prior_box_param);
    return nullptr;
  }
  prior_box_param->max_sizes_size = prior_box_attr->GetMaxSizes().size();
  memcpy(prior_box_param->max_sizes, prior_box_attr->GetMaxSizes().data(),
         prior_box_attr->GetMaxSizes().size() * sizeof(int32_t));
  memcpy(prior_box_param->min_sizes, prior_box_attr->GetMinSizes().data(),
         prior_box_attr->GetMinSizes().size() * sizeof(int32_t));

  if (prior_box_attr->GetAspectRatios().size() > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "PriorBox aspect_ratios size exceeds max num " << MAX_SHAPE_SIZE << ", got "
                  << prior_box_attr->GetAspectRatios();
    free(prior_box_param);
    return nullptr;
  }
  prior_box_param->aspect_ratios_size = prior_box_attr->GetAspectRatios().size();
  memcpy(prior_box_param->aspect_ratios, prior_box_attr->GetAspectRatios().data(),
         prior_box_attr->GetAspectRatios().size() * sizeof(float));
  if (prior_box_attr->GetVariances().size() != COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "PriorBox variances size should be " << COMM_SHAPE_SIZE << ", got "
                  << prior_box_attr->GetVariances().size();
    free(prior_box_param);
    return nullptr;
  }
  memcpy(prior_box_param->variances, prior_box_attr->GetVariances().data(), COMM_SHAPE_SIZE * sizeof(float));
  prior_box_param->flip = prior_box_attr->GetFlip();
  prior_box_param->clip = prior_box_attr->GetClip();
  prior_box_param->offset = prior_box_attr->GetOffset();
  prior_box_param->image_size_h = prior_box_attr->GetImageSizeH();
  prior_box_param->image_size_w = prior_box_attr->GetImageSizeW();
  prior_box_param->step_h = prior_box_attr->GetStepH();
  prior_box_param->step_w = prior_box_attr->GetStepW();
  return reinterpret_cast<OpParameter *>(prior_box_param);
}
Registry PriorBoxParameterRegistry(schema::PrimitiveType_PriorBox, PopulatePriorBoxParameter);

}  // namespace lite
}  // namespace mindspore
