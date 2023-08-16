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
#include "transform/graph_ir/storage_format_convertor.h"

#include "transform/graph_ir/storage_format_config_factory.h"
#include "ir/func_graph.h"

namespace mindspore::transform {
DimsCheckFunc Default4DimsCheckFunc = [](const std::shared_ptr<GeTensorDesc> &desc) -> bool {
  if (desc == nullptr) {
    return false;
  }
  return desc->GetOriginShape().GetDimNum() == 4;
};

REGISTER_STORAGE_FORMAT_CONFIG(Conv2D)
  .set_index_format(0, kOpFormat_NC1HWC0, Default4DimsCheckFunc, "")
  .set_index_format(1, kOpFormat_FRAC_Z, Default4DimsCheckFunc, "");

}  // namespace mindspore::transform
