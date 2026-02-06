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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_STROAGE_FORMAT_CONVERTOR_H
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_STROAGE_FORMAT_CONVERTOR_H

#include <memory>
#include <string>
#include "backend/ge_backend/graph_ir/types.h"
#include "include/utils/utils.h"
#include "mindspore/core/include/ir/manager.h"

namespace mindspore::backend::ge_backend {
extern AnfNodePtr GetMomentumVarByAccum(const AnfNodePtr &node, const NodeUsersMap &node_users);

class StorageFormatConvertor {
 public:
  static bool SetupStorageFormat(const AnfGraphPtr &anf_graph, const AnfNodePtr &param,
                                 const std::shared_ptr<GeTensorDesc> &desc,
                                 const std::string &ori_format = kOpFormat_NCHW);

 private:
  static bool InitParameterKernelInfo(const AnfNodePtr &param, std::string *format);
  static void UpdateParameterKernelInfo(const AnfNodePtr &param, const std::string &format);
  static int32_t GetGeFormat(const AnfNodePtr &src_node, const AnfNodePtr &dst_node, const std::string &storage_format,
                             size_t origin_dim);
  static int32_t GetGeFormat(const AnfNodePtr &src_node, const std::string &storage_format, size_t origin_dim);
  StorageFormatConvertor() = default;
  ~StorageFormatConvertor() = default;
  static void UpdateTensorDesc(const std::shared_ptr<GeTensorDesc> &desc, int32_t format);
  static void SetStorageFormatFromConfig(const AnfGraphPtr &anf_graph, const AnfNodePtr &param,
                                         const std::shared_ptr<GeTensorDesc> &desc);
};
}  // namespace mindspore::backend::ge_backend

#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_STROAGE_FORMAT_CONVERTOR_H
