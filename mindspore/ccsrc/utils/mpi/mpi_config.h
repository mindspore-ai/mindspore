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

#ifndef MINDSPORE_CCSRC_UTILS_MPI_MS_CONTEXT_H_
#define MINDSPORE_CCSRC_UTILS_MPI_MS_CONTEXT_H_
#include <memory>
#include "utils/log_adapter.h"

namespace mindspore {
class MpiConfig {
 public:
  ~MpiConfig() = default;
  MpiConfig(const MpiConfig &) = delete;
  MpiConfig &operator=(const MpiConfig &) = delete;

  static std::shared_ptr<MpiConfig> GetInstance();

  void set_enable_mpi(bool flag) { enable_mpi_ = flag; }
  bool enable_mpi() const { return enable_mpi_; }

 private:
  MpiConfig() : enable_mpi_(false) {}

  static std::shared_ptr<MpiConfig> instance_;
  bool enable_mpi_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_MPI_MS_CONTEXT_H_
