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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_DYNAMIC_UTILS_H
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_DYNAMIC_UTILS_H
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <utility>
#include "include/api/context.h"
#include "include/model.h"
#include "include/transform/graph_ir/types.h"
#include "extendrt/session/lite_graph_executor.h"
#include "common/config_infos.h"

namespace mindspore {
class GeDynamicUtils {
 public:
  static bool IsDynamicInputShapes(const std::vector<ShapeVector> &input_shapes);
  static bool IsDynamicInputShapes(const std::vector<std::pair<std::string, ShapeVector>> &input_shapes);
  static std::vector<std::pair<std::string, ShapeVector>> GetGraphInputShapes(
    const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos, std::string *input_shape);
  static std::vector<std::pair<std::string, ShapeVector>> GetGraphOneRealShapes(
    const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos, std::string *input_shape);
  static std::vector<int64_t> GetDynamicBatchSize(const std::shared_ptr<mindspore::Context> &context,
                                                  const ConfigInfos &config_infos);
  static std::vector<std::vector<int64_t>> GetDynamicImageSize(const std::shared_ptr<mindspore::Context> &context,
                                                               const ConfigInfos &config_infos);
  static std::vector<std::vector<int64_t>> GetDynamicDims(const std::shared_ptr<mindspore::Context> &context,
                                                          const ConfigInfos &config_infos);
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_DYNAMIC_UTILS_H
