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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_INSERT_MEMCPY_ASYNC_FOR_HCCL_OP_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_INSERT_MEMCPY_ASYNC_FOR_HCCL_OP_H_

#include <memory>
#include "pre_activate/common/optimizer.h"
#include "pre_activate/ascend/ascend_helper.h"

namespace mindspore {
namespace opt {
class InsertMemcpyAsyncForHcclOp : public PatternProcessPass {
 public:
  explicit InsertMemcpyAsyncForHcclOp(bool multigraph = true)
      : PatternProcessPass("insert_memcpy_async_for_hccl_op", multigraph),
        kernel_query_(std::make_shared<KernelQuery>()) {}
  ~InsertMemcpyAsyncForHcclOp() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  void InsertMemcpyAsync(const FuncGraphPtr &graph, const CNodePtr &hccl_node) const;
  bool NeedInsertMemcpy(const FuncGraphPtr &graph, const AnfNodePtr &input) const;
  KernelQueryPtr kernel_query_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_INSERT_MEMCPY_ASYNC_FOR_HCCL_OP_H_
