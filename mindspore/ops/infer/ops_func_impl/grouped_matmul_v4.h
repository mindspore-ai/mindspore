/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_GROUPED_MATMUL_V4_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_GROUPED_MATMUL_V4_H_

#include <set>
#include "infer/ops_func_impl/grouped_matmul_base.h"

namespace mindspore {
namespace ops {
class OPS_API GroupedMatmulV4FuncImpl final : public GroupedMatmulBaseFuncImpl {
 public:
  GroupedMatmulV4FuncImpl() {
    idxes_.x = 0;
    idxes_.weight = 1;
    idxes_.split_item_offset = -5;
    idxes_.group_type_offset = -4;
  }
  ~GroupedMatmulV4FuncImpl() = default;

  TypeIdList InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;

 protected:
  void FetchGroupInfo(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;

  int64_t FetchGroupListIndex(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;

  int64_t FetchGroupListSize(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;

 private:
  int64_t group_list_idx_ = 8;
  int64_t group_list_type_idx_ = 14;
  int64_t per_token_scale_idx_ = 7;
  int64_t scale_idx_ = 3;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_GROUPED_MATMUL_V4_H_
