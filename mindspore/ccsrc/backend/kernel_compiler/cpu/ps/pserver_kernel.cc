/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/ps/pserver_kernel.h"

namespace mindspore {
namespace kernel {
namespace ps {
void PServerKernel::Shard(std::vector<size_t> *shape, int axis) {
  MS_EXCEPTION_IF_NULL(shape);
  if ((*shape).size() <= IntToSize(axis)) {
    MS_LOG(EXCEPTION) << "Shape size is invalid.";
  }
  (*shape)[IntToSize(axis)] =
    LongToSize(Util::LocalShard(SizeToLong((*shape)[IntToSize(axis)]), SizeToLong(rank_id_), SizeToLong(pserver_num_)));
}
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore
