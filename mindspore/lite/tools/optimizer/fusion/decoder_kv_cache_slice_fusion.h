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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_DECODER_KV_CACHE_SLICE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_DECODER_KV_CACHE_SLICE_FUSION_H_

#include <string>
#include <set>
#include <unordered_map>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"

namespace mindspore {
namespace opt {
class DecoderKVCacheSliceFusion : public Pass {
 public:
  DecoderKVCacheSliceFusion() : Pass("DecoderKVCacheSliceFusion") {}
  ~DecoderKVCacheSliceFusion() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  int RemoveSliceOp(const FuncGraphManagerPtr &manager, const AnfNodePtr &slice_anf_node,
                    const AnfNodePtr &load_anf_node);

 private:
  std::set<AnfNodePtr> remove_cnode_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_DECODER_KV_CACHE_SLICE_FUSION_H_
