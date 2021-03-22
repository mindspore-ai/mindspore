/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <vector>
#include <string>
#include <map>
#include <set>
#include <stack>
#include <memory>
#include "utils/misc.h"
#include "ir/anf.h"
#include "pipeline/jit/parse/parse_base.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "pipeline/jit/parse/function_block.h"

namespace mindspore {
namespace parse {

// Parse status define
enum ParseStatusCode : int64_t {
  PARSE_SUCCESS = 0,
  PARSE_FUNCTION_IS_NULL,            // python function is null
  PARSE_PARAMETER_INVALID,           // parameter is invalid
  PARSE_NO_RETURN,                   // function no return node
  PARSE_NODE_TYPE_NO_MATCH,          // ast node type is error
  PARSE_NODE_TYPE_UNKNOWN,           // node type is unknown
  PARSE_NODE_METHOD_UNSUPPORTED,     // no method to parse the node
  PARSE_DONT_RESOLVE_SYMBOL,         // can't resolve the string
  PARSE_NOT_SUPPORTED_COMPARE_EXPR,  // the comparison is not supported
  PARSE_FAILURE = 0xFF
};

// max loop count of for statement, when loop count is less then this value, the for loop will be unrolled, otherwise it
//  will be sunk(i.e. not unrolled)
// NOTE: Since when the for loop was unrolled, it depends backend operators `tuple_getitem` and `scalar_add` which were
//  not implemented, so here set MAX_FOR_LOOP_COUNT to int64_t max limit to override default value `600`. This will make
//  the for loop will always be unrolled, but don't worry about the memory were exhausted, an exception will be raised
//  when function call depth exceeds the limit `context.get_context('max_call_depth')`.
const int64_t MAX_FOR_LOOP_COUNT = std::numeric_limits<int64_t>::max();

class AstNodeType;
class ParseAst;

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
  ~LoopContext() { loops_->pop(); }
  const FunctionBlockPtr &EndBlock() const { return loops_->top().end; }

 private:
  std::stack<Loop> *loops_;
};

// Parser to parse python function
class Parser {
 public:
  explicit Parser(const std::shared_ptr<ParseAst> &ast);

  ~Parser() {}
  FuncGraphPtr ParseFuncGraph();
  FuncGraphPtr func_graph() const { return func_graph_; }
  ParseStatusCode errcode() const { return errcode_; }
  std::shared_ptr<ParseAst> ast() const { return ast_; }
  // get location info from the ast node
  LocationPtr GetLocation(const py::object &node) const;
  static void InitParserEnvironment(const py::object &obj);
  static void CleanParserResource();
  static FuncGraphPtr GetTopFuncGraph() { return top_func_graph_.lock(); }
  static void UpdateTopFuncGraph(const FuncGraphPtr &func_graph);

 private:
  // process the stmt node method list
  FunctionBlockPtr ParseReturn(const FunctionBlockPtr &block, const py::object &node);
  // parse expression
  FunctionBlockPtr ParseExpr(const FunctionBlockPtr &block, const py::object &node);
  // process a if statement
  FunctionBlockPtr ParseIf(const FunctionBlockPtr &block, const py::object &node);
  // process a while statement
  FunctionBlockPtr ParseWhile(const FunctionBlockPtr &block, const py::object &node);
  // process a for statement
  FunctionBlockPtr ParseFor(const FunctionBlockPtr &block, const py::object &node);
  FunctionBlockPtr ParseForIter(const FunctionBlockPtr &block, const py::object &node);
  FunctionBlockPtr ParseForLoop(const FunctionBlockPtr &block, const py::object &node);
  // process a function def statement
  FunctionBlockPtr ParseFunctionDef(const FunctionBlockPtr &block, const py::object &node);
  // process a augment assign
  FunctionBlockPtr ParseAugAssign(const FunctionBlockPtr &block, const py::object &node);
  // process a global declaration
  FunctionBlockPtr ParseGlobal(const FunctionBlockPtr &block, const py::object &node);
  // process assign statement
  FunctionBlockPtr ParseAssign(const FunctionBlockPtr &block, const py::object &node);
  // process break statement
  FunctionBlockPtr ParseBreak(const FunctionBlockPtr &block, const py::object &node);
  // process continue statement
  FunctionBlockPtr ParseContinue(const FunctionBlockPtr &block, const py::object &node);
  // process pass statement
  FunctionBlockPtr ParsePass(const FunctionBlockPtr &block, const py::object &node);
  // process the expr and slice node method list
  AnfNodePtr ParseBinOp(const FunctionBlockPtr &block, const py::object &node);
  // process a variable name
  AnfNodePtr ParseName(const FunctionBlockPtr &block, const py::object &node);
  // process NoneType
  AnfNodePtr ParseNone(const FunctionBlockPtr &block, const py::object &node);
  // process Ellipsis
  AnfNodePtr ParseEllipsis(const FunctionBlockPtr &block, const py::object &node);
  // process a integer or float number
  AnfNodePtr ParseNum(const FunctionBlockPtr &block, const py::object &node);
  // process a string variable
  AnfNodePtr ParseStr(const FunctionBlockPtr &block, const py::object &node);
  // process a Constant
  AnfNodePtr ParseConstant(const FunctionBlockPtr &block, const py::object &node);
  // process a name
  AnfNodePtr ParseNameConstant(const FunctionBlockPtr &block, const py::object &node);
  // process a function call
  AnfNodePtr ParseCall(const FunctionBlockPtr &block, const py::object &node);
  // process function 'super'
  AnfNodePtr ParseSuper(const FunctionBlockPtr &block, const py::list &args);
  // process the if expression
  AnfNodePtr ParseIfExp(const FunctionBlockPtr &block, const py::object &node);
  // process class type define
  AnfNodePtr ParseAttribute(const FunctionBlockPtr &block, const py::object &node);
  // process a compare expression
  AnfNodePtr ParseCompare(const FunctionBlockPtr &block, const py::object &node);
  // process a bool operation
  AnfNodePtr ParseBoolOp(const FunctionBlockPtr &block, const py::object &node);
  // process a lambda operation
  AnfNodePtr ParseLambda(const FunctionBlockPtr &block, const py::object &node);
  // process a tuple
  AnfNodePtr ParseTuple(const FunctionBlockPtr &block, const py::object &node);
  // process a tuple
  AnfNodePtr ParseList(const FunctionBlockPtr &block, const py::object &node);
  // process a tuple
  AnfNodePtr ParseSubscript(const FunctionBlockPtr &block, const py::object &node);
  // process a slice
  AnfNodePtr ParseSlice(const FunctionBlockPtr &block, const py::object &node);

  // process a extslice
  AnfNodePtr ParseExtSlice(const FunctionBlockPtr &block, const py::object &node);

  // process a tuple
  AnfNodePtr ParseIndex(const FunctionBlockPtr &block, const py::object &node);

  // process a unaryop
  AnfNodePtr ParseUnaryOp(const FunctionBlockPtr &block, const py::object &node);

  // process a dict ast node expression
  AnfNodePtr ParseDict(const FunctionBlockPtr &block, const py::object &node);
  // generate argument nodes for ast  function node
  void GenerateArgsNodeForFunction(const FunctionBlockPtr &block, const py::object &function_node);
  // generate argument default value for ast  function node
  void GenerateArgsDefaultValueForFunction(const FunctionBlockPtr &block, const py::object &function_node);
  // parse ast function node
  FunctionBlockPtr ParseFunction(const py::object &function_node, const FunctionBlockPtr &block = nullptr);
  // parse ast statements
  FunctionBlockPtr ParseStatements(FunctionBlockPtr block, const py::object &stmt_node);
  // parse one ast statement node
  FunctionBlockPtr ParseStatement(const FunctionBlockPtr &block, const py::object &node);
  // parse an ast expression node
  AnfNodePtr ParseExprNode(const FunctionBlockPtr &block, const py::object &node);

  void MakeConditionBlocks(const FunctionBlockPtr &block, const FunctionBlockPtr &trueBlock,
                           const FunctionBlockPtr &falseBlock);
  void RemoveUnnecessaryPhis();
  // write a new var
  void WriteAssignVars(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &value_node);

  // assign value to single variable name
  void HandleAssignName(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &assigned_node);

  // assign value to tuple
  void HandleAssignTuple(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &assigned_node);

  // assign value to class member
  void HandleAssignClassMember(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &assigned_node);

  // assign value to subscript
  void HandleAssignSubscript(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &assigned_node);

  // process a bool operation value list
  AnfNodePtr ProcessBoolOpValueList(const FunctionBlockPtr &block, const py::list &value_list, AstSubType mode);

  CNodePtr GenerateIteratorInFor(const FunctionBlockPtr &block, const pybind11::object &node,
                                 const AnfNodePtr &op_iter);

  CNodePtr GenerateCondInFor(const ParameterPtr &iter_param, const FunctionBlockPtr &header_block,
                             const AnfNodePtr &op_hasnext);

  FunctionBlockPtr GenerateBlockInFor(const TraceInfoPtr &trace_info);

  bool ParseKeywordsInCall(const FunctionBlockPtr &block, const py::object &node,
                           std::vector<AnfNodePtr> *packed_arguments);

  bool ParseArgsInCall(const FunctionBlockPtr &block, const py::list &args, std::vector<AnfNodePtr> *packed_arguments,
                       std::vector<AnfNodePtr> *group_arguments);

  AnfNodePtr GenerateAnfNodeForCall(const FunctionBlockPtr &block, const AnfNodePtr &call_function_anf_node,
                                    const std::vector<AnfNodePtr> &packed_arguments,
                                    const std::vector<AnfNodePtr> &group_arguments, bool need_unpack) const;
  ScopePtr GetScopeForParseFunction();
  void BuildMethodMap();
  FunctionBlockPtr MakeFunctionBlock(const Parser &parse) {
    FunctionBlockPtr block = std::make_shared<FunctionBlock>(parse);
    // In order to keep effect order in the sub-graphs which generated by control flow.
    // We copy the flags from the top graph to the sub-graphs.
    if (func_graph_ && !func_graph_->attrs().empty()) {
      block->func_graph()->set_attrs(func_graph_->attrs());
    }
    func_block_list_.push_back(block);
    return block;
  }
  // return a make tuple for input elements list
  AnfNodePtr GenerateMakeTuple(const FunctionBlockPtr &block, const std::vector<AnfNodePtr> &element_nodes);

  // shared_ptr will be hold by GraphManager, so just hold a weak ref here.
  static FuncGraphWeakPtr top_func_graph_;
  // Python function id, used to indicate whether two CNodes come from the same Python function
  const std::shared_ptr<ParseAst> &ast_;
  FuncGraphPtr func_graph_;
  // error code setwhen parsing ast tree
  ParseStatusCode errcode_;

  // hold all reference for FunctionBlock in this round of parsing,
  // so in FunctionBlock class we can use FunctionBlock* in member
  // pre_blocks_ and jumps_ to break reference cycle.
  std::vector<FunctionBlockPtr> func_block_list_;
  using pStmtFunc = FunctionBlockPtr (Parser::*)(const FunctionBlockPtr &block, const py::object &node);
  using pExprFunc = AnfNodePtr (Parser::*)(const FunctionBlockPtr &block, const py::object &node);
  // define the function map to parse ast Statement
  std::map<std::string, pStmtFunc> stmt_method_map_;
  // define the function map to parse ast expression
  std::map<std::string, pExprFunc> expr_method_map_;
  // Save current loops to support 'continue', 'break' statement.
  std::stack<Loop> loops_;
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
class ParseAst {
 public:
  explicit ParseAst(const py::object &obj) : obj_(obj), target_type_(PARSE_TARGET_UNKNOW), function_line_offset_(-1) {}

  ~ParseAst() = default;

  bool InitParseAstInfo(const std::string &python_mod_get_parse_method = PYTHON_MOD_GET_PARSE_METHOD);

  py::object GetAstNode();

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

 private:
  // save obj,eg: class instance or function
  py::object obj_;

  // function or class method.
  py::function function_;

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

// update the graph flags
bool UpdateFuncGraphFlags(const py::object &obj, const FuncGraphPtr &func_graph);

AnfNodePtr GetMixedPrecisionCastHelp(const FuncGraphPtr &func_graph, const AnfNodePtr &param);
TypePtr GetMixedPrecisionTargetType(const FuncGraphPtr &func_graph);

}  // namespace parse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_PARSE_H_
