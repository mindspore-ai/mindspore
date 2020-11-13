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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SLICE_PREPOSE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SLICE_PREPOSE_PASS_H_

#include <vector>
#include <memory>
#include <utility>
#include "tools/converter/converter_flags.h"
#include "backend/optimizer/common/pass.h"
#include "include/errorcode.h"
#include "mindspore/core/ir/manager.h"
#include "schema/inner/model_generated.h"

using mindspore::lite::converter::FmkType;
namespace mindspore::opt {
using lite::RET_ERROR;
using lite::RET_OK;
using lite::STATUS;
using TransactionPtr = std::shared_ptr<mindspore::FuncGraphTransaction>;
using NodeUsedListPtr = std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>>;
class SlicePreposePass : public Pass {
 public:
  SlicePreposePass() : Pass("slice_prepose_pass") {}
  ~SlicePreposePass() override = default;
  bool Run(const FuncGraphPtr &graph) override;
  void SetFmkType(FmkType fmkType) { this->fmk_type = fmkType; }

 private:
  schema::SliceT *GetSliceT(const CNodePtr &cnode);
  bool DoPrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &preceed_cnode);
  STATUS SwapSliceWithPreceed(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &preceed_cnode,
                              const int index, const TransactionPtr &tr = nullptr);
  bool PreposeWithSoftmax(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &softmax_cnode);

 private:
  FmkType fmk_type = lite::converter::FmkType_ONNX;
};
}  // namespace mindspore::opt

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SLICE_PREPOSE_PASS_H_
