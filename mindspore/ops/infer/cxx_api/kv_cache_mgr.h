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

#ifndef MINDSPORE_CORE_OPS_KV_CACHE_MGR_H_
#define MINDSPORE_CORE_OPS_KV_CACHE_MGR_H_
#include <vector>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore::ops {
constexpr auto kNameKVCacheMgr = "KVCacheMgr";
/// \brief Define KVCacheMgr operator prototype.
class OPS_API KVCacheMgr : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(KVCacheMgr);
  /// \brief Constructor.
  KVCacheMgr() : BaseOperator(kNameKVCacheMgr) { InitIOName({"past, cur, index"}, {"past"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] shifts
  void Init() const;
};
}  // namespace mindspore::ops

#endif  // MINDSPORE_CORE_OPS_KV_CACHE_MGR_H_
