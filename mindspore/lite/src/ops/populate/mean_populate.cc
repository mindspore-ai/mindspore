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

#include "src/ops/mean.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/reduce_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateMeanParameter(const mindspore::lite::PrimitiveC *primitive) {
  ReduceParameter *mean_param = reinterpret_cast<ReduceParameter *>(malloc(sizeof(ReduceParameter)));
  if (mean_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReduceParameter failed.";
    return nullptr;
  }
  memset(mean_param, 0, sizeof(ReduceParameter));
  mean_param->op_parameter_.type_ = primitive->Type();
  auto mean = reinterpret_cast<mindspore::lite::Mean *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  mean_param->keep_dims_ = mean->GetKeepDims();
  auto axisVector = mean->GetAxis();
  if (axisVector.size() > REDUCE_MAX_AXES_NUM) {
    MS_LOG(ERROR) << "Reduce axes size " << axisVector.size() << " exceed limit " << REDUCE_MAX_AXES_NUM;
    free(mean_param);
    return nullptr;
  }
  mean_param->num_axes_ = static_cast<int>(axisVector.size());
  int i = 0;
  for (auto iter = axisVector.begin(); iter != axisVector.end(); iter++) {
    mean_param->axes_[i++] = *iter;
  }
  mean_param->mode_ = static_cast<int>(schema::ReduceMode_ReduceMean);
  return reinterpret_cast<OpParameter *>(mean_param);
}
Registry MeanParameterRegistry(schema::PrimitiveType_Mean, PopulateMeanParameter);

}  // namespace lite
}  // namespace mindspore
