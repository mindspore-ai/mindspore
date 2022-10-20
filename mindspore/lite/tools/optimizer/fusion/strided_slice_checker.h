/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_STRIDED_SLICE_CHECKER_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_STRIDED_SLICE_CHECKER_H_

#include <vector>
#include "ir/anf.h"
#include "tools/lite_exporter/fetch_content.h"

namespace mindspore {
namespace opt {
class StridedSliceChecker {
 public:
  StridedSliceChecker() = default;
  ~StridedSliceChecker() = default;
  static bool CheckCommonInfo(const CNodePtr &strided_slice);
  static int GetBegin(const CNodePtr &strided_slice, std::vector<int> *begin);
  static int GetEnd(const CNodePtr &strided_slice, std::vector<int> *end);

 private:
  static bool CheckStepIsOne(const CNodePtr &strided_slice);
  static int GetConstTensor(const CNodePtr &strided_slice, size_t index, lite::DataInfo *data_info);
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_STRIDED_SLICE_CHECKER_H_
