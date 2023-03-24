/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_PARSE_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_PARSE_H_

#include <limits>
#include <utility>
#include <tuple>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <stack>
#include <memory>
#include "utils/misc.h"
#include "ir/anf.h"
#include "pipeline/jit/parse/parse_base.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/parse/function_block.h"

namespace mindspore {
namespace parse {
// Parse status define
enum ParseStatusCode : int64_t {
  PARSE_SUCCESS = 0,
  PARSE_FUNCTION_IS_NULL,            // Python function is null
  PARSE_PARAMETER_INVALID,           // Parameter is invalid
  PARSE_NO_RETURN,                   // Function no return node
  PARSE_NODE_TYPE_NO_MATCH,          // Ast node type is error
  PARSE_NODE_TYPE_UNKNOWN,           // Node type is unknown
  PARSE_NODE_METHOD_UNSUPPORTED,     // No method to parse the node
  PARSE_DONT_RESOLVE_SYMBOL,         // Can't resolve the string
  PARSE_NOT_SUPPORTED_COMPARE_EXPR,  // The comparison is not supported
  PARSE_FAILURE = 0xFF
};

constexpr char kStandardMethodModelName[] = "mindspore._extends.parse.standard_method";
constexpr char kMsLenWithCheck[] = "ms_len_with_iterable_check";
constexpr char kMsLenWithListCompCheck[] = "ms_next_with_dyn_input_check";

// Max loop count of for statement, when loop count is less then this value, the for loop will be unrolled, otherwise it
// will be sunk(i.e. not unrolled)
// NOTE: Since when the for loop was unrolled, it depends backend operators `tuple_getitem` and `scalar_add` which were
//  not implemented, so here set MAX_FOR_LOOP_COUNT to int64_t max limit to override default value `600`. This will make
//  the for loop will always be unrolled, but don't worry about the memory were exhausted, an exception will be raised
//  when function call depth exceeds the limit `context.get_context('max_call_depth')`.
const int64_t MAX_FOR_LOOP_COUNT = std::numeric_limits<int64_t>::max();

class AstNodeType;
class ParseFunctionAst;

// Save loop info for 'continue' and 'break' statements.
struct Loop {
  // Loop header block.
  FunctionBlockPtr header;
  // Loop iterator node, used in 'for loop'.
  AnfNodePtr iterator;
  // Loop end block.
  FunctionBlockPtr end;

  Loop(const FunctionBlockPtr &header, const AnfNodePtr &iterator, const FunctionBlockPtr &end)
      : header(header), iterator(iterator), end(end) {}
  ~Loop() = default;
};

// Loop context for loop stack management.
class LoopContext {
 public:
  LoopContext(std::stack<Loop> *loops, const FunctionBlockPtr &header, const AnfNodePtr &iterator) : loops_(loops) {
    loops_->emplace(header, iterator, nullptr);
  }
  ~LoopContext() {
    try {
      MS_EXCEPTION_IF_NULL(loops_);
      loops_->pop();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Exception when pop. Error info " << e.what();
    } catch (...) {
      MS_LOG(ERROR) << "Throw exception when pop.";
    }
    loops_ = nullptr;
  }

  const FunctionBlockPtr &EndBlock() const { return loops_->top().end; }

 private:
  std::stack<Loop> *loops_;
};

struct ArgsContext {
  bool need_unpack{false};
  bool has_interpret_without_internal{false};
  bool has_interpret_internal{false};

  std::vector<AnfNodePtr> packed_arguments;
  std::vector<AnfNodePtr> group_arguments;
  ArgsContext() {}
  ~ArgsContext() = default;
};

// Parser to parse python function
class Parser {
 public:
  explicit Parser(const std::shared_ptr<ParseFunctionAst> &ast);

  ~Parser() {}
  FuncGraphPtr ParseFuncGraph();
  FuncGraphPtr func_graph() const { return func_graph_; }
  ParseStatusCode errcode() const { return errcode_; }
  std::shared_ptr<ParseFunctionAst> ast() const { return ast_; }
  const std::string &support_fallback() const { return support_fallback_; }
  // Get location info from the ast node
  LocationPtr GetLocation(const py::object &node) const;
  static void InitParserEnvironment(const py::object &obj);
  static void CleanParserResource();
  static FuncGraphPtr GetTopFuncGraph() { return top_func_graph_.lock(); }
  static void UpdateTopFuncGraph(const FuncGraphPtr &func_graph);

 private:
  // Process the stmt node method list
  FunctionBlockPtr ParseReturn(const FunctionBlockPtr &block, const py::object &node);
  // Parse expression
  FunctionBlockPtr ParseExpr(const FunctionBlockPtr &block, const py::object &node);
  // Process a if statement
  FunctionBlockPtr ParseIf(const FunctionBlockPtr &block, const py::object &node);
  // Process a while statement
  FunctionBlockPtr ParseWhile(const FunctionBlockPtr &block, const py::object &node);
  // Process a for statement
  FunctionBlockPtr ParseFor(const FunctionBlockPtr &block, const py::object &node);
  FunctionBlockPtr ParseForUnroll(const FunctionBlockPtr &block, const py::object &node);
  FunctionBlockPtr ParseForRepeat(const FunctionBlockPtr &block, const py::object &node);
  // Process a function def statement
  FunctionBlockPtr ParseFunctionDef(const FunctionBlockPtr &block, const py::object &node);
  // Process a augment assign
  FunctionBlockPtr ParseAugAssign(const FunctionBlockPtr &block, const py::object &node);
  // Process a global declaration
  FunctionBlockPtr ParseGlobal(const FunctionBlockPtr &block, const py::object &node);
  // Process assign statement
  FunctionBlockPtr ParseAssign(const FunctionBlockPtr &block, const py::object &node);
  // Process break statement
  FunctionBlockPtr ParseBreak(const FunctionBlockPtr &block, const py::object &node);
  // Process continue statement
  FunctionBlockPtr ParseContinue(const FunctionBlockPtr &block, const py::object &node);
  // Process pass statement
  FunctionBlockPtr ParsePass(const FunctionBlockPtr &block, const py::object &node);
  // Process raise statement
  FunctionBlockPtr ParseRaise(const FunctionBlockPtr &block, const py::object &node);
  // Process assert statement
  FunctionBlockPtr ParseAssert(const FunctionBlockPtr &block, const py::object &node);
  // Process with statement
  FunctionBlockPtr ParseWith(const FunctionBlockPtr &block, const py::object &node);

  // Process withitem
  AnfNodePtr ParseWithitem(const FunctionBlockPtr &block, const py::object &node, const AnfNodePtr &context_expr_node);
  // Process the expr and slice node method list
  AnfNodePtr ParseBinOp(const FunctionBlockPtr &block, const py::object &node);
  // Process a variable name
  AnfNodePtr ParseName(const FunctionBlockPtr &block, const py::object &node);
  // Process NoneType
  AnfNodePtr ParseNone(const FunctionBlockPtr &, const py::object &);
  // Process Ellipsis
  AnfNodePtr ParseEllipsis(const FunctionBlockPtr &, const py::object &);
  // Process an integer or float number
  AnfNodePtr ParseNum(const FunctionBlockPtr &, const py::object &node);
  // Process a string variable
  AnfNodePtr ParseStr(const FunctionBlockPtr &, const py::object &node);
  // Process a Constant
  AnfNodePtr ParseConstant(const FunctionBlockPtr &, const py::object &node);
  // Process a name
  AnfNodePtr ParseNameConstant(const FunctionBlockPtr &, const py::object &node);
  // Process a function call
  AnfNodePtr ParseCall(const FunctionBlockPtr &block, const py::object &node);
  // Process function 'super'
  AnfNodePtr ParseSuper(const FunctionBlockPtr &block, const py::list &args);
  // Process the if expression
  AnfNodePtr ParseIfExp(const FunctionBlockPtr &block, const py::object &node);
  // Process class type define
  AnfNodePtr ParseAttribute(const FunctionBlockPtr &block, const py::object &node);
  // Process ms Tensor
  AnfNodePtr ParseMsTensor(const FunctionBlockPtr &block, const py::object &node, const py::object &value_body,
                           const AnfNodePtr &value_node);
  // Process dtype._null
  AnfNodePtr ParseNull(const FunctionBlockPtr &block, const py::object &value_body) const;
  // Process a compare expression
  AnfNodePtr ParseCompare(const FunctionBlockPtr &block, const py::object &node);
  // Process a bool operation
  AnfNodePtr ParseBoolOp(const FunctionBlockPtr &block, const py::object &node);
  // Process a lambda operation
  AnfNodePtr ParseLambda(const FunctionBlockPtr &block, const py::object &node);
  // Process a tuple
  AnfNodePtr ParseTuple(const FunctionBlockPtr &block, const py::object &node);
  // Process a tuple
  AnfNodePtr ParseList(const FunctionBlockPtr &block, const py::object &node);
  // Process a tuple
  AnfNodePtr ParseSubscript(const FunctionBlockPtr &block, const py::object &node);
  // Process a slice
  AnfNodePtr ParseSlice(const FunctionBlockPtr &block, const py::object &node);
  // Process a extslice
  AnfNodePtr ParseExtSlice(const FunctionBlockPtr &block, const py::object &node);
  // Process a tuple
  AnfNodePtr ParseIndex(const FunctionBlockPtr &block, const py::object &node);
  // Process a unaryop
  AnfNodePtr ParseUnaryOp(const FunctionBlockPtr &block, const py::object &node);
  // Process a dict ast node expression
  AnfNodePtr ParseDictByKeysAndValues(const FunctionBlockPtr &block, const std::vector<AnfNodePtr> &key_nodes,
                                      const std::vector<AnfNodePtr> &value_nodes);
  AnfNodePtr ParseDict(const FunctionBlockPtr &block, const py::object &node);
  // Process ListComp expression
  AnfNodePtr ParseListComp(const FunctionBlockPtr &block, const py::object &node);
  FunctionBlockPtr ParseListCompIter(const FunctionBlockPtr &block, const py::object &node,
                                     const py::object &generator_node);
  AnfNodePtr ParseListCompIfs(const FunctionBlockPtr &list_body_block, const ParameterPtr &list_param,
                              const py::object &node, const py::object &generator_node);
  AnfNodePtr ParseJoinedStr(const FunctionBlockPtr &block, const py::object &node);
  AnfNodePtr ParseFormattedValue(const FunctionBlockPtr &block, const py::object &node);
  std::vector<AnfNodePtr> ParseException(const FunctionBlockPtr &block, const py::list &args, const std::string &name);
  std::vector<AnfNodePtr> ParseRaiseCall(const FunctionBlockPtr &block, const py::object &node);
  void ParseStrInError(const FunctionBlockPtr &block, const py::list &args, std::vector<AnfNodePtr> *str_nodes);
  FunctionBlockPtr MakeAssertErrorBlock(const FunctionBlockPtr &block, const py::object &node);
  AnfNodePtr ProcessAttributeWithClassMember(const FunctionBlockPtr &block, const py::object &node) const;

  // Transform tail call to parallel call.
  void TransformParallelCall();
  void LiftRolledBodyGraphFV();
  void LiftIfBranchGraphFV();

  // Check if script_text is in global/local params.
  bool IsScriptInParams(const std::string &script_text, const py::dict &global_dict,
                        const std::map<std::string, AnfNodePtr> &local_keys, const FuncGraphPtr &func_graph) const;
  // Set the interpret flag for the node calling the interpret node.
  void UpdateInterpretForUserNode(const AnfNodePtr &user_node, const AnfNodePtr &node) const;
  void UpdateInterpretForUserNode(const AnfNodePtr &user_node, const std::vector<AnfNodePtr> &nodes) const;
  // Make interpret node.
  AnfNodePtr MakeInterpretNode(const FunctionBlockPtr &block, const AnfNodePtr &value_node, const string &script_text);
  // Convert interpret iter node to list.
  AnfNodePtr ConvertInterpretIterNodeToList(const FunctionBlockPtr &block, const AnfNodePtr &iter_node,
                                            const py::object &iter_obj);
  // Check if the node need interpreting.
  AnfNodePtr HandleInterpret(const FunctionBlockPtr &block, const AnfNodePtr &value_node,
                             const py::object &value_object);
  bool CheckNeedConvertInterpret(const FunctionBlockPtr &block, const AnfNodePtr &node,
                                 const string &script_text) const;
  // Handle interpret for augassign expression.
  AnfNodePtr HandleInterpretForAugassign(const FunctionBlockPtr &block, const AnfNodePtr &augassign_node,
                                         const py::object &op_object, const py::object &target_object,
                                         const py::object &value_object);

  // Generate argument nodes for ast function node
  void GenerateArgsNodeForFunction(const FunctionBlockPtr &block, const py::object &fn_node);
  // Generate argument default value for ast function node
  void GenerateArgsDefaultValueForFunction(const FunctionBlockPtr &block, const py::object &fn_node);
  // Parse ast function node
  FunctionBlockPtr ParseDefFunction(const py::object &node, const FunctionBlockPtr &block = nullptr);
  // Parse lambda function node
  FunctionBlockPtr ParseLambdaFunction(const py::object &node, const FunctionBlockPtr &block = nullptr);
  // Parse ast statements
  FunctionBlockPtr ParseStatements(const FunctionBlockPtr &block, const py::object &nodes);
  // Parse one ast statement node
  FunctionBlockPtr ParseStatement(const FunctionBlockPtr &block, const py::object &node);
  // Parse an ast expression node
  AnfNodePtr ParseExprNode(const FunctionBlockPtr &block, const py::object &node);

  void MakeConditionBlocks(const FunctionBlockPtr &pre_block, const FunctionBlockPtr &true_block,
                           const FunctionBlockPtr &false_block) const;
  std::shared_ptr<std::map<ParameterPtr, AnfNodePtr>> CalRemovablePhis();
  void CreatePhiArgMaps(std::map<ParameterPtr, std::set<AnfNodePtr>> *phi_to_args,
                        std::map<AnfNodePtr, std::set<ParameterPtr>> *arg_to_phis);
  static void PrintPhiArgMaps(const std::map<ParameterPtr, std::set<AnfNodePtr>> &phi_to_args,
                              const std::map<AnfNodePtr, std::set<ParameterPtr>> &arg_to_phis);
  static void UpdatePhiArgMapsRepeatedly(std::map<ParameterPtr, std::set<AnfNodePtr>> *phi_to_args,
                                         std::map<AnfNodePtr, std::set<ParameterPtr>> *arg_to_phis);
  static std::shared_ptr<std::map<ParameterPtr, AnfNodePtr>> CollectRemovablePhiArgs(
    const std::map<ParameterPtr, std::set<AnfNodePtr>> &phi_to_args);
  void RemoveUnnecessaryPhis();
  // Write a new var
  void WriteAssignVars(const FunctionBlockPtr &block, const py::object &target_object, const AnfNodePtr &value_node);

  // Assign value to single variable name
  void HandleAssignName(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &assigned_node) const;

  // Assign value to tuple
  void HandleAssignTuple(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &assigned_node);

  // Assign value to class member
  void HandleAssignClassMember(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &assigned_node);

  // Assign value to subscript
  void HandleAssignSubscript(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &assigned_node);

  // Interpret the return node.
  AnfNodePtr HandelReturnExprNode(const FunctionBlockPtr &block, const AnfNodePtr &return_expr_node,
                                  const py::object &value_object);

  // Process a bool operation value list
  AnfNodePtr ProcessBoolOpValueList(const FunctionBlockPtr &block, const py::list &value_list, AstSubType mode);

  FunctionBlockPtr GenerateBlock(const TraceInfoPtr &trace_info);

  void ParseKeywordsInCall(const FunctionBlockPtr &block, const py::object &node, ArgsContext *args_context);

  void ParseArgsInCall(const FunctionBlockPtr &block, const py::list &args, ArgsContext *args_context);
  AnfNodePtr GenerateAnfNodeForCall(const FunctionBlockPtr &block, const AnfNodePtr &call_function_node,
                                    const ArgsContext &args_context) const;
  ScopePtr GetScopeForParseFunction();
  // Check the value is subscript is reference type
  bool IsSubscriptReferenceType(const py::object &obj);
  void BuildMethodMap();
  FunctionBlockPtr MakeFunctionBlock(const Parser &parse) {
    FunctionBlockPtr block = std::make_shared<FunctionBlock>(parse);
    // In order to keep effect order in the sub-graphs which generated by control flow.
    // We copy the flags from the top graph to the sub-graphs.
    if (func_graph_ && !func_graph_->attrs().empty()) {
      for (const auto &attr : func_graph_->attrs()) {
        // The flag FUNC_GRAPH_OUTPUT_NO_RECOMPUTE should be only set in the top graph.
        if (attr.first != FUNC_GRAPH_OUTPUT_NO_RECOMPUTE) {
          block->func_graph()->set_attr(attr.first, attr.second);
        }
      }
    }
    func_block_list_.push_back(block);
    return block;
  }
  // Return a make tuple for input elements list
  AnfNodePtr GenerateMakeTuple(const FunctionBlockPtr &block, const std::vector<AnfNodePtr> &element_nodes);
  // Check if the node is pop operation.
  bool IsPopOperation(const AnfNodePtr &node) const;
  // Check if branch block contains break/continue/return statement, and propagate that flag back to block.
  void CheckControlFlowAlterationInIf(std::pair<FunctionBlockPtr, FunctionBlockPtr> *branch_graphs_pair,
                                      const FunctionBlockPtr &branch_block, const FunctionBlockPtr &branch_end,
                                      const FunctionBlockPtr &after_block, const FunctionBlockPtr &block) const;
  // Check if body block contains return statement, and propagate that flag back to block.
  void CheckReturnInLoop(const FunctionBlockPtr &block, const FunctionBlockPtr &body_block) const;

  // Check whether the functions referred by this function and itself are missing 'return' statement.
  void CheckFuncReturn(const FuncGraphPtr &fn);

  // The shared_ptr will be hold by GraphManager, so just hold a weak ref here.
  static FuncGraphWeakPtr top_func_graph_;
  // Python function id, used to indicate whether two CNodes come from the same Python function
  const std::shared_ptr<ParseFunctionAst> &ast_;
  FuncGraphPtr func_graph_;
  // Error code setwhen parsing ast tree
  ParseStatusCode errcode_;
  py::object list_pop_target_obj_;

  // Hold all reference for FunctionBlock in this round of parsing,
  // so in FunctionBlock class we can use FunctionBlock* in member
  // pre_blocks_ and jumps_ to break reference cycle.
  std::vector<FunctionBlockPtr> func_block_list_;
  using StmtFunc = FunctionBlockPtr (Parser::*)(const FunctionBlockPtr &block, const py::object &node);
  using ExprFunc = AnfNodePtr (Parser::*)(const FunctionBlockPtr &block, const py::object &node);
  // Define the function map to parse ast Statement
  std::map<std::string, StmtFunc> stmt_method_map_;
  // Define the function map to parse ast expression
  std::map<std::string, ExprFunc> expr_method_map_;
  // Save current loops to support 'continue', 'break' statement.
  std::stack<Loop> loops_;
  string support_fallback_;

  // The func graphs to transform tail call ir to independent call ir.
  // Contains: {former_graph, middle_graph}, latter_graph is no need.
  std::vector<std::vector<std::pair<FunctionBlockPtr, FunctionBlockPtr>>> parallel_call_graphs_;
  // The true branch and false branch call info. of if statement.
  std::vector<std::tuple<CNodePtr, FunctionBlockPtr, FunctionBlockPtr>> if_branch_calls_;
  // The rolled_body callers info. for later lifting operation.
  std::vector<std::pair<CNodePtr, FunctionBlockPtr>> rolled_body_calls_;
};

// AST node type define code to ast
class AstNodeType {
 public:
  AstNodeType(const py::object &node, const std::string &name, AstMainType type)
      : node_(node), node_name_(name), main_type_(type) {}

  ~AstNodeType() {}

  std::string node_name() const { return node_name_; }

  py::object node() const { return node_; }

  AstMainType main_type() const { return main_type_; }

 private:
  const py::object &node_;
  const std::string node_name_;
  AstMainType main_type_;
};

using AstNodeTypePtr = std::shared_ptr<AstNodeType>;

// A helper class to parse python function
class ParseFunctionAst {
 public:
  explicit ParseFunctionAst(const py::object &obj)
      : obj_(obj), target_type_(PARSE_TARGET_UNKNOW), function_line_offset_(-1) {}

  ~ParseFunctionAst() = default;

  bool InitParseAstInfo(const std::string &python_mod_get_parse_method = PYTHON_MOD_GET_PARSE_METHOD);

  py::object GetAstNode();

  py::str GetAstNodeText(const py::object &node);

  py::list GetArgs(const py::object &func_node);

  py::list GetArgsDefaultValues(const py::object &func_node);

  AstNodeTypePtr GetNodeType(const py::object &node);

  AstSubType GetOpType(const py::object &node);

  template <class... T>
  py::object CallParserObjMethod(const std::string &method, const T &... args) {
    return python_adapter::CallPyObjMethod(parser_, method, args...);
  }

  template <class... T>
  py::object CallParseModFunction(const std::string &function, const T &... args) {
    return python_adapter::CallPyModFn(module_, function, args...);
  }

  const std::string &function_name() const { return function_name_; }

  const std::string &function_module() const { return function_module_; }

  const std::string &function_filename() const { return function_filename_; }

  int64_t function_line_offset() const { return function_line_offset_; }

  py::function function() { return function_; }

  ParseTargetTypeDef target_type() const { return target_type_; }

  py::object obj() { return obj_; }

  py::object parser() { return parser_; }

  py::object module() { return module_; }

  py::object ast_tree() { return ast_tree_; }

  bool IsClassMember(const py::object &node);
  bool IsClassMemberRecursive(const py::object &node);

 private:
  // Save obj,eg: class instance or function
  py::object obj_;

  // Function or class method.
  py::function function_;

  py::object ast_tokens_;
  py::object ast_tree_;
  py::object parser_;
  py::module module_;

  // Is function or method
  ParseTargetTypeDef target_type_;

  std::string function_name_;
  std::string function_module_;
  std::string function_filename_;
  int64_t function_line_offset_;
};

// Update the graph flags
bool UpdateFuncGraphFlags(const py::object &obj, const FuncGraphPtr &func_graph);

AnfNodePtr GetMixedPrecisionCastHelp(const FuncGraphPtr &func_graph, const AnfNodePtr &param);
}  // namespace parse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_PARSE_H_
