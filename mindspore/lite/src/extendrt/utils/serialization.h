/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRD_EXTENDRT_UTILS_SERIALIZATION_H_
#define MINDSPORE_LITE_SRD_EXTENDRT_UTILS_SERIALIZATION_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "include/api/status.h"
#include "include/api/types.h"
#include "include/api/model.h"
#include "include/api/graph.h"
#include "include/api/dual_abi_helper.h"
#include "mindspore/core/base/base.h"

namespace mindspore::infer {
class Serialization {
 public:
  static mindspore::Status Load(const void *model_data, size_t data_size, mindspore::ModelType model_type,
                                mindspore::Graph *graph, const mindspore::Key &dec_key = {},
                                const std::string &dec_mode = kDecModeAesGcm, const std::string &mindir_path = "");

 private:
  static mindspore::FuncGraphPtr ConvertStreamToFuncGraph(const char *buf, const size_t buf_size, bool is_lite,
                                                          const std::string &mindir_path = "");
};
}  // namespace mindspore::infer
#endif  // MINDSPORE_LITE_SRD_EXTENDRT_UTILS_SERIALIZATION_H_
