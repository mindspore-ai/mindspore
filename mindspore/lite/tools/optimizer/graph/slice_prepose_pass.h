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
#include "include/backend/optimizer/pass.h"
#include "include/errorcode.h"
#include "mindspore/core/ir/manager.h"
#include "include/registry/converter_context.h"

using mindspore::converter::FmkType;
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
  static void ClearCNodeAbstractValue(const CNodePtr &cnode);
  static STATUS SwapSliceWithPreceed(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                     const CNodePtr &preceed_cnode, int index, const TransactionPtr &tr = nullptr);
  static ValueNodePtr CreateSliceValueNode(const std::vector<int64_t> &axes);
  static ValueNodePtr CopySliceValueNode(const CNodePtr &slice_cnode);
  static CNodePtr InsertSlice(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &inputs,
                              const CNodePtr &preceed_cnode, int index, const TransactionPtr &tr);
  static STATUS VerifySliceAttrs(const CNodePtr &slice_cnode, int dim = -1);
  static STATUS SliceParamDeBroadcast(const CNodePtr &slice_cnode, const std::vector<int64_t> &ref_shape,
                                      std::vector<int64_t> *axes, std::vector<int> *begin, std::vector<int> *size);
  static CNodePtr CreateReshapeCNode(const FuncGraphPtr &graph, const std::vector<int64_t> &shape,
                                     const AbstractBasePtr &abstract, const CNodePtr &preceed_cnode);
  static bool SiblingsAreSameSlice(const NodeUsedListPtr &output_node_list, const std::vector<int64_t> &ref_shape = {});
  static int64_t GetReshapeAbnormalAxeIn(const std::vector<int64_t> &shape_in, const std::vector<int64_t> &shape_out,
                                         std::vector<int64_t> *mapped_axe);
  static int64_t GetReshapeAbnormalIndexOut(const CNodePtr &slice_cnode, const std::vector<int64_t> &mapped_axe,
                                            const std::vector<int64_t> &shape_out, std::vector<int64_t> *shape_out_copy,
                                            bool *is_normal_mode, bool *support_abnormal_mode);
  static bool PreposeWithNormalReshape(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                       const CNodePtr &reshape_cnode, const std::vector<int64_t> &shape_in,
                                       const std::vector<int64_t> &shape_out_copy,
                                       const std::vector<int64_t> &mapped_axe);
  static CNodePtr CreateSlice1ForReshapePrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                                const CNodePtr &matmul_cnode, const std::vector<int64_t> &shape_in,
                                                int64_t abnormal_axe_in, int64_t count_sliced_axe_in,
                                                bool slice_at_front);
  static CNodePtr CreateSlice2ForReshapePrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                                const CNodePtr &new_reshape1_cnode,
                                                const std::vector<int64_t> &new_shape1, int64_t abnormal_axe_in,
                                                int64_t count_sliced2, bool slice_at_front);
  static bool PreposeWithAbnormalReshape(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                         const CNodePtr &matmul_cnode, const std::vector<int64_t> &shape_in,
                                         const std::vector<int64_t> &shape_out, int64_t abnormal_axe_in,
                                         int64_t abnormal_index_out);
  static bool GetArithmeticInputInfo(const CNodePtr &arithmetic_cnode, std::vector<AnfNodePtr> *inputs,
                                     std::vector<std::vector<int64_t>> *shapes, std::vector<bool> *is_default_params);

  static bool DoPrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &preceed_cnode);

  static bool PreposeWithSoftmax(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &softmax_cnode);
  static bool PreposeWithReshape(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &reshape_cnode);
  static bool PreposeWithMatmul(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &matmul_cnode);
  static bool PreposeWithFullConnection(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                        const CNodePtr &fc_cnode);
  static bool PreposeWithTranspose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                   const CNodePtr &transpose_cnode);
  static bool PreposeWithArithmetic(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                    const CNodePtr &arithmetic_cnode);
  static bool MergeSequentialSlice(const FuncGraphPtr &graph, const CNodePtr &slice1_cnode,
                                   const CNodePtr &slice2_cnode);
  static bool MergeParallelSlice(const FuncGraphPtr &graph, const NodeUsedListPtr &slices);

 private:
  FmkType fmk_type = converter::kFmkTypeOnnx;
};
}  // namespace mindspore::opt

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SLICE_PREPOSE_PASS_H_
