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
#include "nnacl/slice_parameter.h"
#include "ops/fusion/slice_fusion.h"
using mindspore::ops::kNameSliceFusion;
using mindspore::schema::PrimitiveType_SliceFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateSliceOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<SliceParameter *>(PopulateOpParameter<SliceParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new SliceParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::SliceFusion *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not TopKFusion.";
    return nullptr;
  }

  auto axes = op->get_axes();
  // if begin is not const input, then axis can not be decided in converter
  if (axes.size() != 0) {
    if (axes.size() > DIMENSION_8D) {
      MS_LOG(ERROR) << "Invalid axes size: " << axes.size();
      free(param);
      return nullptr;
    }
    for (size_t i = 0; i < axes.size(); ++i) {
      auto id = axes[i];
      if (id > INT32_MAX) {
        MS_LOG(ERROR) << "Invalid axes: " << id;
        free(param);
        return nullptr;
      }
      param->axis_[i] = id;
    }
  } else {
    // use default axes
    for (int32_t i = 0; i < DIMENSION_8D; i++) {
      param->axis_[i] = i;
    }
  }
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameSliceFusion, PrimitiveType_SliceFusion, PopulateSliceOpParameter)
}  // namespace lite
}  // namespace mindspore
