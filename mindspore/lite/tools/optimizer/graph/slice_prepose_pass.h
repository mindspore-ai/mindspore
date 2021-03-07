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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SLICE_PREPOSE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SLICE_PREPOSE_PASS_H_

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include "tools/converter/converter_flags.h"
#include "backend/optimizer/common/pass.h"
#include "include/errorcode.h"
#include "mindspore/core/ir/manager.h"

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
  void ClearCNodeAbstractValue(const CNodePtr &cnode);
  STATUS SwapSliceWithPreceed(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &preceed_cnode,
                              const int index, const TransactionPtr &tr = nullptr);
  ValueNodePtr CreateSliceValueNode(const FuncGraphPtr &graph, const std::vector<int64_t> &axes);
  ValueNodePtr CopySliceValueNode(const FuncGraphPtr &graph, const CNodePtr &slice_cnode);
  CNodePtr InsertSlice(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &inputs, const CNodePtr &preceed_cnode,
                       const int index, const TransactionPtr &tr);
  STATUS VerifySliceAttrs(const CNodePtr &slice_cnode, const int dim = -1);
  STATUS SliceParamDeBroadcast(const CNodePtr &slice_cnode, const std::vector<int64_t> &ref_shape,
                               std::vector<int64_t> *axes, std::vector<int> *begin, std::vector<int> *size);
  CNodePtr CreateReshapeCNode(const FuncGraphPtr &graph, const std::vector<int64_t> &shape,
                              const AbstractBasePtr &abstract, const CNodePtr &preceed_cnode);
  bool SiblingsAreSameSlice(const FuncGraphPtr &graph, const NodeUsedListPtr &output_node_list,
                            const std::vector<int64_t> &ref_shape = {});
  int64_t GetReshapeAbnormalAxeIn(const std::vector<int64_t> &shape_in, const std::vector<int64_t> &shape_out,
                                  std::vector<int64_t> *mapped_axe);
  int64_t GetReshapeAbnormalIndexOut(const CNodePtr &slice_cnode, const std::vector<int64_t> &mapped_axe,
                                     const std::vector<int64_t> &shape_out, std::vector<int64_t> *shape_out_copy,
                                     bool *is_normal_mode, bool *support_abnormal_mode);
  bool PreposeWithNormalReshape(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &reshape_cnode,
                                const std::vector<int64_t> &shape_in, const std::vector<int64_t> &shape_out_copy,
                                const std::vector<int64_t> &mapped_axe);
  CNodePtr CreateSlice1ForReshapePrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                         const CNodePtr &matmul_cnode, const std::vector<int64_t> &shape_in,
                                         const int64_t abnormal_axe_in, const int64_t count_sliced_axe_in,
                                         const bool slice_at_front);
  CNodePtr CreateSlice2ForReshapePrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                         const CNodePtr &new_reshape1_cnode, const std::vector<int64_t> &new_shape1,
                                         const int64_t abnormal_axe_in, const int64_t count_sliced_axe_in,
                                         const int64_t count_sliced2, const bool slice_at_front);
  bool PreposeWithAbnormalReshape(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &reshape_cnode,
                                  const CNodePtr &matmul_cnode, const std::vector<int64_t> &shape_in,
                                  const std::vector<int64_t> &shape_out, const int64_t abnormal_axe_in,
                                  const int64_t abnormal_index_out);
  bool GetArithmeticInputInfo(const CNodePtr &arithmetic_cnode, std::vector<AnfNodePtr> *inputs,
                              std::vector<std::vector<int64_t>> *shapes, std::vector<bool> *is_default_params);

  bool DoPrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &preceed_cnode);

  bool PreposeWithSoftmax(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &softmax_cnode);
  bool PreposeWithReshape(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &reshape_cnode);
  bool PreposeWithMatmul(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &matmul_cnode);
  bool PreposeWithFullConnection(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &fc_cnode);
  bool PreposeWithTranspose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &transpose_cnode);
  bool PreposeWithArithmetic(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &arithmetic_cnode);
  bool MergeSequentialSlice(const FuncGraphPtr &graph, const CNodePtr &slice1_cnode, const CNodePtr &slice2_cnode);
  bool MergeParallelSlice(const FuncGraphPtr &graph, const NodeUsedListPtr &slices);

 private:
  FmkType fmk_type = lite::converter::FmkType_ONNX;
};
}  // namespace mindspore::opt

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SLICE_PREPOSE_PASS_H_
