/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_FUNCTIONAL_OVERLOAD_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_FUNCTIONAL_OVERLOAD_H_

#include <map>
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/meta_func_graph.h"
#include "ops/op_def.h"

namespace mindspore {
namespace prim {
class DeprecatedTensorMethod : public MetaFuncGraph {
 public:
  explicit DeprecatedTensorMethod(const std::string &name, const std::string &method)
      : MetaFuncGraph(name), method_(method) {}
  ~DeprecatedTensorMethod() override = default;
  MS_DECLARE_PARENT(DeprecatedTensorMethod, MetaFuncGraph)
  std::string method() const { return method_; }
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &) override;

 private:
  std::string method_;
};
using DeprecatedTensorMethodPtr = std::shared_ptr<DeprecatedTensorMethod>;

struct PrimitiveAttr {
  const std::string &prim_name;
  const ops::OpDefPtr &op_def;
  bool has_varargs = false;
  size_t varargs_index = 0;
  bool is_match_args_size = false;
};

class PrimitiveConverter {
 public:
  explicit PrimitiveConverter(const std::string &functional_name,
                              const abstract::AbstractBasePtrList &input_args_abs_list, const bool is_method,
                              bool *need_pack)
      : functional_name_(functional_name),
        input_args_abs_list_(input_args_abs_list),
        is_method_(is_method),
        need_pack_(need_pack) {}
  ValuePtr Convert();

 private:
  bool MatchPrimitiveArgs(PrimitiveAttr *cur_prim);
  size_t GetPrimDefaultSize(const std::vector<ops::OpInputArg> &expect_op_args, const std::string &prim_name,
                            size_t varargs_index, bool has_varargs) const;
  bool CheckArgsSize(PrimitiveAttr *cur_prim);
  bool CheckKwargs(PrimitiveAttr *cur_prim);
  bool CheckPositionArgs(PrimitiveAttr *cur_prim);
  std::string BuildMatchInfo(const std::vector<std::string> &arg_info_list) const;
  std::string BuildDetailedErrorMsg(const std::vector<std::string> &arg_info_list) const;
  void GetOpDtypeList();
  void PrintErrorMessages();
  bool CheckExplicitSequence(PrimitiveAttr *cur_prim, const std::vector<ops::OpInputArg> &expect_op_args) const;
  bool CheckImplicitTuple(PrimitiveAttr *cur_prim, const std::vector<ops::OpInputArg> &expect_op_args);
  const std::string &functional_name_;
  const abstract::AbstractBasePtrList &input_args_abs_list_;
  std::vector<ops::OP_DTYPE> input_position_args_dtype_;
  std::map<std::string, ops::OP_DTYPE> input_keyword_args_dtype_;
  const bool is_method_;
  bool *need_pack_;
  std::vector<size_t> match_index_;
  std::vector<std::string> error_msgs_;
  size_t first_failed_position_ = 0;
  bool is_keyword_ = false;
  size_t prim_list_size_ = 0;  // Some of the tensor_method_overload_signature_map has no deprecated
  bool has_deprecated_ = false;
};

bool IsFunctionalMethod(const TypeId &type_id, const std::string &method_name);
std::map<size_t, std::pair<ValuePtr, bool>> &GetFunctionalConvertCache();
std::string BuildArgsTypeString(const TypePtr &arg_abs);
AnfNodePtr ConvertFunctionalToPrimitive(const std::string &functional_name, const AnfNodePtrList &inputs_list,
                                        const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode,
                                        const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func,
                                        bool is_method);
AnfNodePtr ConvertFunctionalToPyExecute(const FunctionalPtr &functional, const AnfNodePtrList &inputs_list,
                                        const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode,
                                        bool is_method);
}  // namespace prim
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_FUNCTIONAL_OVERLOAD_H_
