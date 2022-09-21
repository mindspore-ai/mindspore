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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_OPERATOR_INFO_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_OPERATOR_INFO_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "tools/optimizer/parallel/split_strategy.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"

namespace mindspore {
namespace opt {
/**
 * Do following steps to make a operator support parallel:
 *
 * 1.Add the schema::PrimitiveType_XXX to ParallelPass::PARALLEL_LIST;
 * 2.Add a pair of type and string name to ParallelPass::type_string;
 * 3.Implement a class XXXInfo whose parent is OperatorInfo;
 *    3.1.Override CheckStrategy(), InferParallelCNodes() and InferReplaceOp()
 * 4.include header file of XXXInfo in ops_info_head_files.h
 * 5.REGISTER XXXInfo in dynamic_creator.cc
 */
using schema::ReduceMode;
class OperatorInfo;
using OperatorInfoPtr = std::shared_ptr<OperatorInfo>;
class OperatorInfo {
 public:
  OperatorInfo(const std::string &name, const SplitStrategy &strategy)
      : name_(std::move(name)),
        strategy_(std::move(strategy)),
        replace_op_(nullptr),
        func_graph_(nullptr),
        cnode_(nullptr) {}
  virtual ~OperatorInfo() = default;
  const std::string name() const { return name_; }
  void set_name(const std::string &name) { name_ = name; }
  void Init(const FuncGraphPtr &func_graph, const CNodePtr &cnode, int32_t fmk_type);
  int DoSplit();
  AnfNodePtr replace_op() const { return replace_op_; }

 protected:
  int CheckSplitResult(const AnfNodePtr &anf_node, const std::vector<AnfNodePtr> &split_results, int target_output_num);

  int CreateMultipleOutputsOfAnfNode(const AnfNodePtr &node, size_t output_num, std::vector<AnfNodePtr> *outputs);

  AnfNodePtr CreateConcateNode(const CNodePtr &orig_node, const std::vector<AnfNodePtr> &input_nodes,
                               int32_t concat_dim, size_t input_nodes_num);
  AnfNodePtr CreateReduceNode(const CNodePtr &orig_node, const std::vector<AnfNodePtr> &input_nodes,
                              size_t input_nodes_num);

  std::shared_ptr<abstract::AbstractTensor> CreateFakeAbstractTensor() const;

  virtual AnfNodePtr CreateOutputsOfSplit(const CNodePtr &input_node, size_t input_index,
                                          std::vector<AnfNodePtr> *split_outputs, size_t split_dim, size_t split_num,
                                          const std::vector<int64_t> &splits) = 0;
  virtual int InferReplaceOp() = 0;
  virtual int InferParallelCNodes() = 0;
  virtual int CheckStrategy(const SplitStrategy &strategy) = 0;

 protected:
  std::string name_;
  SplitStrategy strategy_;
  AnfNodePtr replace_op_{nullptr};
  std::vector<AnfNodePtr> parallel_output_nodes_;
  FuncGraphPtr func_graph_{nullptr};
  CNodePtr cnode_{nullptr};
  int32_t fmk_type_{};
  TypeId operator_type_id_ = kNumberTypeFloat32;

 private:
  int SetCNodeBackend();
  int CheckStrategyValue();
};

// a template func for normal op_coder creator
template <typename T>
std::unique_ptr<OperatorInfo> OperatorInfoCreator(const std::string &name, const SplitStrategy &strategy) {
  std::unique_ptr<T> coder = std::make_unique<T>(name, strategy);
  return coder;
}

bool is_any_none(const std::vector<int64_t> &split);
bool is_any_not_none(const std::vector<int64_t> &split);

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_PARALLEL_OPERATOR_INFO_H_
