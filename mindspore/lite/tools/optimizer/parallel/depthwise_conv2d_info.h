/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_PASS_PARALLEL_DEPTHWISE_CONV2D_INFO_H_
#define MINDSPORE_LITE_SRC_PASS_PARALLEL_DEPTHWISE_CONV2D_INFO_H_
#include <string>
#include <vector>
#include <memory>
#include "tools/optimizer/parallel/split_strategy.h"
#include "tools/optimizer/parallel/conv2d_info.h"
#include "ops/fusion/conv2d_fusion.h"
#include "include/errorcode.h"
namespace mindspore {
namespace opt {
class DepthwiseConv2DInfo : public Conv2DInfo {
 public:
  DepthwiseConv2DInfo(const std::string &name, const SplitStrategy &strategy) : Conv2DInfo(name, strategy) {}
  ~DepthwiseConv2DInfo() override = default;

 private:
  int InferReplaceOp() override;
  int InferParallelCNodes() override;
  int CheckStrategy(const SplitStrategy &strategy) override;
  int ConstructOutputCNodes(const std::shared_ptr<ops::Conv2DFusion> &conv_prim,
                            const std::vector<AnfNodePtr> &feature_split_outputs,
                            const std::vector<AnfNodePtr> &kernel_split_outputs,
                            const std::vector<AnfNodePtr> &bias_split_outputs) override;
  AnfNodePtr CreateOutputsOfSplit(const CNodePtr &orig_node, size_t input_index, std::vector<AnfNodePtr> *split_outputs,
                                  size_t split_num, const std::vector<int64_t> &splits);
  int CreateConstantOutputsOfSplit(std::vector<AnfNodePtr> *split_outputs, int input_index);

  int CheckDepthWiseConv2DPrimitiveType();

  bool CheckSplitOutputs(const std::vector<AnfNodePtr> &feature_split_outputs,
                         const std::vector<AnfNodePtr> &kernel_split_outputs,
                         const std::vector<AnfNodePtr> &bias_split_outputs);

  void AdJustConvPrim(const std::shared_ptr<ops::Conv2DFusion> &conv_prim, int64_t *visited_in_channel,
                      int64_t *visited_out_channel, int64_t *visited_group, int output_conv_index);

  void AdJustInputs(const std::shared_ptr<ops::Conv2DFusion> &conv_prim,
                    const std::vector<AnfNodePtr> &feature_split_outputs,
                    const std::vector<AnfNodePtr> &kernel_split_outputs,
                    const std::vector<AnfNodePtr> &bias_split_outputs, int output_conv_index);

 private:
  int split_dim_{0};
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_PASS_PARALLEL_DEPTHWISE_CONV2D_INFO_H_
