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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PS_PSERVER_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PS_PSERVER_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/kernel.h"
#include "ps/util.h"

namespace mindspore {
namespace kernel {
namespace ps {
using mindspore::ps::Util;
class PServerKernel {
 public:
  PServerKernel(size_t rank_id, size_t pserver_num, size_t worker_num)
      : rank_id_(rank_id), pserver_num_(pserver_num), worker_num_(worker_num) {}
  ~PServerKernel() = default;
  PServerKernel(const PServerKernel &) = delete;
  PServerKernel &operator=(const PServerKernel &) = delete;
  virtual void InitKernel(const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &) {}
  virtual void InitKernel(const CNodePtr &cnode,
                          const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &) {}
  virtual void ReInit(const std::vector<std::vector<size_t>> &) {}
  virtual bool Execute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                       const std::vector<AddressPtr> &outputs) = 0;
  virtual void UpdateEmbeddings(float *embedding_table, const size_t *lookup_ids, const float *update_vals,
                                size_t ids_size) {}
  virtual const std::vector<size_t> &input_sizes() const = 0;
  virtual const std::vector<size_t> &output_sizes() const = 0;
  virtual const std::vector<size_t> &workspace_sizes() const = 0;

 protected:
  virtual void ReInit(const std::vector<AddressPtr> &) {}
  void Shard(std::vector<size_t> *shape, int axis);

  size_t rank_id_;
  size_t pserver_num_;
  size_t worker_num_;
};
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PS_PSERVER_KERNEL_H_
