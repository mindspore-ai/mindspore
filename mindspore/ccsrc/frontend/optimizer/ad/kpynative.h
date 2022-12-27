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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_KPYNATIVE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_KPYNATIVE_H_

#include <memory>
#include <vector>
#include "ir/anf.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace ad {
class KPynativeCell {
 public:
  virtual ~KPynativeCell() = default;
  virtual void UpdateOutputNodeOfTopCell(const AnfNodePtr &output_node, const ValuePtr &sen_out) = 0;
  // Grad for cell which may have user passed front propagate FuncGraph.
  // c_node: CNode with contains the construct function graph of cell  (index 0) and the formal input parameters of that
  // cell. op_args: the arguments list of each input parameters.
  // out: the op result.
  // fprop_fg: user defined back propagate cnode which output is the bprop_fg.
  //           Should have prototype: (sens_input1, sens_input2, ...) bprop_fg(input1, input2, ..., out, dout)
  virtual bool KPynativeWithFProp(const CNodePtr &c_node, const ValuePtrList &op_args, const ValuePtr &out,
                                  const FuncGraphPtr &fprop_fg) = 0;
};

using KPynativeCellPtr = std::shared_ptr<KPynativeCell>;

struct GradAttr {
  bool grad_all_inputs;
  bool grad_weights;
  bool has_sens;
  bool get_by_position;
  bool weight_param_is_tuple;

  GradAttr(bool get_all, bool get_by_list, bool sens_param, bool get_by_position, bool weight_param_is_tuple)
      : grad_all_inputs(get_all),
        grad_weights(get_by_list),
        has_sens(sens_param),
        get_by_position(get_by_position),
        weight_param_is_tuple(weight_param_is_tuple) {}
};

// bprop_fg: user defined back propagate funcgraph or back propagate funcgraph of primitive, it will be passed after
//           just parsed. will have prototype:
//           (sens_input1, sens_input2, ...) bprop_fg(input1, input2, ..., out, dout)
// c_node: CNode with contains the prim (index 0) and the formal input parameters of that prim.
// op_args: the arguments list of each input parameters.
// out: the op result.
// return: the returned funcgraph should have the same prototype.
FuncGraphPtr OptimizeBPropFuncGraph(const FuncGraphPtr &bprop_fg, const CNodePtr &cnode, const ValuePtrList &op_args,
                                    const ValuePtr &out);

// Start building back propagate funcgraph for this cell.
// cell_inputs: the input parameter list of this cell except the weights;
KPynativeCellPtr GradPynativeCellBegin(const AnfNodePtrList &cell_inputs,
                                       const std::vector<ValuePtr> &input_param_values);

// Return the back propagate funcgraph for this cell.
// weights: weights parameters used in this cell.
// grad_inputs: return sensitivity for input parameters;
// grad_weights: return sensitivity for weights;
// has_sens_arg: caller will pass sens args;
// return: the returned funcgraph will have prototype:
// if has_sens_arg is true
// (sens_input1, sens_input2, ..., sens_weight0, sens_weight1, ) bprop_fg(input1, input2, ..., weight0, weight1, ...,
// sens_out)
// else:
// (sens_input1, sens_input2, ..., sens_weight0, sens_weight1, ) bprop_fg(input1, input2, ..., weight0, weight1, ...)
// if build_formal_param is true
// each cnode in primal funcgraph is replaced by formal cnode
// else:
// each cnode in primal funcgraph is replaced by value node
FuncGraphPtr GradPynativeCellEnd(const KPynativeCellPtr &k_cell, const AnfNodePtrList &weights,
                                 const std::vector<size_t> &grad_position, const GradAttr &grad_attr,
                                 bool build_formal_param = false);

// Grad for each operation.
// c_node: CNode with contains the prim (index 0) and the formal input parameters of that prim.
// op_args: the arguments list of each input parameters.
// out: the op result.
bool GradPynativeOp(const KPynativeCellPtr &k_cell, const CNodePtr &cnode, const ValuePtrList &op_args,
                    const ValuePtr &out);

// Grad for cell which may have user defined back propagate function.
// c_node: CNode with contains the construct function graph of cell  (index 0) and the formal input parameters of that
// cell. op_args: the arguments list of each input parameters.
// out: the op result.
// bprop_fg: user defined back propagate funcgraph, it should be passed after just parsed.
//           Should have prototype: (sens_input1, sens_input2, ...) bprop_fg(input1, input2, ..., out, dout)
bool GradPynativeWithBProp(const KPynativeCellPtr &k_cell, const CNodePtr &c_node, const ValuePtrList &op_args,
                           const ValuePtr &out, const FuncGraphPtr &bprop_fg);

// Clear all static resources that used in grad process
void ClearKPynativeCellStaticRes();
}  // namespace ad
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_GRAD_H_
