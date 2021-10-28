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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_MULTI_CONV_INFO_H
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_MULTI_CONV_INFO_H
#include <memory>
#include <vector>
#include "tools/optimizer/parallel/multi_node_split.h"
#include "tools/optimizer/fisson/fisson_util.h"
#include "ops/fusion/conv2d_fusion.h"
namespace mindspore {
namespace opt {
class MultiConvSplit : public MultiNodeSplit {
 public:
  explicit MultiConvSplit(const SplitStrategy &strategy, PrimitiveType primitive_type, int32_t fmk_type = -1,
                          int32_t num = 3)
      : MultiNodeSplit(), strategy_(strategy), primitive_type_(primitive_type), fmk_type_(fmk_type), num_(num) {}

  ~MultiConvSplit() = default;

  AnfNodePtr DoSplit(const FuncGraphPtr &func_graph, const AnfNodePtr &node) override;

 protected:
  bool CheckSplitValid();

  virtual AnfNodePtr SplitMultiConv(const AnfNodePtr &node) = 0;

  virtual void AdJustConvPrim(const std::shared_ptr<ops::Conv2DFusion> &ori_attr, const ShapeVector &input_shape,
                              int output_conv_index) = 0;

  virtual AnfNodePtr MultiConvNHSplit(const AnfNodePtr &node);

  virtual void AdJustInputs(const AnfNodePtr &ori_node, const std::vector<AnfNodePtr> &new_inputs_node,
                            int output_conv_index, std::vector<AnfNodePtr> *conv_inputs);

  virtual bool CreateNewConvNode(const AnfNodePtr &ori_conv_node, const std::vector<AnfNodePtr> &conv_inputs,
                                 int output_conv_index, std::vector<AnfNodePtr> *outputs_node);

  virtual bool SplitSingleConv(const AnfNodePtr &ori_node, const std::vector<AnfNodePtr> &inputs_node,
                               std::vector<AnfNodePtr> *outputs_node);

 protected:
  FuncGraphPtr func_graph_{nullptr};
  SplitInfo split_info_{};
  SplitStrategy strategy_{};
  PrimitiveType primitive_type_{schema::PrimitiveType_NONE};
  int32_t fmk_type_{-1};
  int32_t num_{0};
  std::vector<AnfNodePtr> conv_nodes_{};

 private:
  int GenSplitInfo();
  int GetMultiConvNodes(const AnfNodePtr &conv_node);

 private:
  std::vector<int64_t> ori_split_ratios_{};
};

class MultiConvSplitN final : public MultiConvSplit {
 public:
  MultiConvSplitN(const SplitStrategy &strategy, PrimitiveType primitive_type, int32_t fmk_type = -1, int32_t num = 3)
      : MultiConvSplit(strategy, primitive_type, fmk_type, num) {}
  ~MultiConvSplitN() = default;
  AnfNodePtr SplitMultiConv(const AnfNodePtr &node) override;

  void AdJustConvPrim(const std::shared_ptr<ops::Conv2DFusion> &ori_attr, const ShapeVector &input_shape,
                      int output_conv_index) override {}
};

class MultiConvSplitCIN final : public MultiConvSplit {
 public:
  MultiConvSplitCIN(const SplitStrategy &strategy, PrimitiveType primitive_type, int32_t fmk_type = -1, int32_t num = 3)
      : MultiConvSplit(strategy, primitive_type, fmk_type, num) {}
  ~MultiConvSplitCIN() = default;
  AnfNodePtr SplitMultiConv(const AnfNodePtr &node) override;

  void AdJustConvPrim(const std::shared_ptr<ops::Conv2DFusion> &ori_attr, const ShapeVector &input_shape,
                      int output_conv_index) override {}
};

class MultiConvSplitCOUT final : public MultiConvSplit {
 public:
  MultiConvSplitCOUT(const SplitStrategy &strategy, PrimitiveType primitive_type, int32_t fmk_type = -1,
                     int32_t num = 3)
      : MultiConvSplit(strategy, primitive_type, fmk_type, num) {}
  ~MultiConvSplitCOUT() = default;
  AnfNodePtr SplitMultiConv(const AnfNodePtr &node) override;

  void AdJustConvPrim(const std::shared_ptr<ops::Conv2DFusion> &ori_attr, const ShapeVector &input_shape,
                      int output_conv_index) override {}
};

class MultiConvSplitH final : public MultiConvSplit {
 public:
  MultiConvSplitH(const SplitStrategy &strategy, PrimitiveType primitive_type, int32_t fmk_type = -1, int32_t num = 3)
      : MultiConvSplit(strategy, primitive_type, fmk_type, num) {}
  ~MultiConvSplitH() = default;
  AnfNodePtr SplitMultiConv(const AnfNodePtr &node) override;

  void AdJustConvPrim(const std::shared_ptr<ops::Conv2DFusion> &ori_attr, const ShapeVector &input_shape,
                      int output_conv_index) override;
};

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_MULTI_CONV_INFO_H
