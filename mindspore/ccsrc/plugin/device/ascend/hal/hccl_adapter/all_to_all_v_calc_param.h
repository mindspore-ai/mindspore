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
#ifndef MINDSPORE_RUNTIME_HCCL_ADAPTER_ALL_TO_ALL_V_CALC_PARAM_H
#define MINDSPORE_RUNTIME_HCCL_ADAPTER_ALL_TO_ALL_V_CALC_PARAM_H

#include <memory>
#include <vector>
#include <string>
#include "mindspore/core/ir/anf.h"

namespace mindspore::hccl {
class AllToAllvCalcParam {
 public:
  AllToAllvCalcParam(const CNodeWeakPtr &cnode, uint32_t rank_size);
  ~AllToAllvCalcParam() = default;
  void CalcOpParam();

  const std::vector<int64_t> &GetSendCounts() const { return send_counts_; }
  const std::vector<int64_t> &GetSendDispls() const { return sdispls_; }
  const std::vector<int64_t> &GetRecvCounts() const { return recv_counts_; }
  const std::vector<int64_t> &GetRecvDispls() const { return rdispls_; }

 private:
  void CalcMemOffset(const std::vector<size_t> &mem_sizes, const std::vector<size_t> &real_sizes,
                     const std::string &rank_ids_attr, std::vector<int64_t> *counts, std::vector<int64_t> *displs);
  CNodeWeakPtr node_;
  uint32_t rank_size_;
  std::vector<int64_t> send_counts_;
  std::vector<int64_t> sdispls_;
  std::vector<int64_t> recv_counts_;
  std::vector<int64_t> rdispls_;
};
}  // namespace mindspore::hccl
#endif  // MINDSPORE_RUNTIME_HCCL_ADAPTER_ALL_TO_ALL_V_CALC_PARAM_H
