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
#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_SPLIT_MODEL_ASCEND_CCE_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_SPLIT_MODEL_ASCEND_CCE_H_

#include <set>
#include <string>
#include <memory>
#include <vector>
#include "tools/graph_kernel/converter/split_model_ascend.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
namespace mindspore::graphkernel::inner {
class SplitModelAscendCCE : public SplitModelAscend {
 public:
  SplitModelAscendCCE() { InitDefaultAreaOps(); }
  virtual ~SplitModelAscendCCE() = default;

 protected:
  AreaMode GetDefaultAreaMode(const PrimOpPtr &) const override;
  void InitFusePatterns() override;
  void InitDefaultAreaOps();
  std::set<std::string> default_area_op_;
  using FusePatterns = std::vector<std::shared_ptr<FusePattern>>;
  void AddCceOpPattern(std::string &&pattern_name, FusePatterns &&patterns);
};
}  // namespace mindspore::graphkernel::inner
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_SPLIT_MODEL_ASCEND_CCE_H_
