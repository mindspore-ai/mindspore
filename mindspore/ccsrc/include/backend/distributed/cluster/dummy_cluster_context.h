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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_DUMMY_CLUSTER_CONTEXT_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_DUMMY_CLUSTER_CONTEXT_H_

#include <map>
#include <set>
#include <string>
#include <memory>
#include <atomic>
#include <vector>
#include "include/backend/distributed/constants.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace distributed {
namespace cluster {
// The dummy cluster context interface. This class is for ut test and windows compiling.
class BACKEND_EXPORT ClusterContext {
 public:
  ~ClusterContext() = default;
  DISABLE_COPY_AND_ASSIGN(ClusterContext)
  static std::shared_ptr<ClusterContext> instance();

  bool Initialize() const;
  bool Finalize(uint32_t timeout = kDefaultFinishTimeout) const;
  std::string node_role() const;
  uint32_t node_num(const std::string &node_role);
  bool initialized() const;
  void set_cluster_exit_with_exception();
  bool cluster_exit_with_exception() const;

 private:
  ClusterContext() = default;
};
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_DUMMY_CLUSTER_CONTEXT_H_
