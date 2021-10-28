/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef _WIN32
#include <dirent.h>
#endif
#include <memory>
#include <string>
#include <utility>
#include "ir/anf.h"
#include "pybind_api/ir/primitive_py.h"
#include "ir/meta_func_graph.h"
#include "ir/func_graph_cloner.h"
#include "ir/manager.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/parse/parse.h"
#include "pipeline/jit/parse/resolve.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/composite.h"
#include "utils/utils.h"
#include "utils/symbolic.h"
#include "utils/primitive_utils.h"
#include "utils/ms_context.h"
#include "utils/info.h"
#include "debug/trace.h"
#include "debug/common.h"
#include "debug/dump_proto.h"
#include "mindspore/core/load_mindir/load_model.h"
#include "utils/system/sha256.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace ad {
KPrim g_k_prims;

namespace {
constexpr char kBpropMindIRSuffix[] = "_bprop.mindir";
constexpr char kBpropMindIRDir[] = "/../bprop_mindir/";
constexpr char serializable_bprop_ops[] = "serializable_bprop_ops";
constexpr char bprop_mindir_module[] = "mindspore.ops.bprop_mindir";

#ifndef _WIN32
std::string GetBpropDir() {
  static std::string bprop_dir("");
  if (bprop_dir.empty()) {
    py::module mod = py::module::import("mindspore.ops._grad");
    auto grad_file_path = mod.attr("__file__").cast<std::string>();
    bprop_dir = grad_file_path.substr(0, grad_file_path.find_last_of('/'));
  }
  return bprop_dir;
}

bool BpropMindirDirExists() {
  auto bprop_mindir_dir = GetBpropDir() + kBpropMindIRDir;
  DIR *dir = opendir(bprop_mindir_dir.c_str());
  if (dir != nullptr) {
    if (closedir(dir) == -1) {
      MS_LOG(WARNING) << "The bprop mindir dir \"" << bprop_mindir_dir << "\" close failed!";
    }
    return true;
  }
  MS_LOG(INFO) << "The bprop mindir dir \"" << bprop_mindir_dir << "\" doesn't exists.";
  return false;
}

// Get the serializable bprop list from the module mindspore.ops.bprop_mindir in python.
std::unordered_set<std::string> GetSerializableBpropList() {
  std::unordered_set<std::string> serializable_bprop_list;
  if (!BpropMindirDirExists()) {
    return serializable_bprop_list;
  }
  py::module mod = py::module::import(bprop_mindir_module);
  py::object serializable_bprop_ops_attr = mod.attr(serializable_bprop_ops);
  if (!py::isinstance<py::list>(serializable_bprop_ops_attr)) {
    MS_LOG(WARNING) << "Can not get the the serializable bprop ops list from python, it is not a python list.";
    return serializable_bprop_list;
  }

  auto ops_list = serializable_bprop_ops_attr.cast<py::list>();
  for (size_t i = 0; i < ops_list.size(); ++i) {
    auto prim_adapter = ops_list[i].cast<PrimitivePyAdapterPtr>();
    if (prim_adapter == nullptr) {
      MS_LOG(EXCEPTION) << "The python obj in serializable bprop list should be a Primitive, but it is "
                        << py::str(ops_list[i]);
    }
    (void)serializable_bprop_list.insert(prim_adapter->name());
  }
  return serializable_bprop_list;
}

bool IsSerializableBprop(const std::string &prim_name) {
  static std::unordered_set<std::string> serializable_bprop_list = GetSerializableBpropList();
  return std::any_of(serializable_bprop_list.begin(), serializable_bprop_list.end(),
                     [&prim_name](const std::string &serializable_bprop_prim_name) {
                       return prim_name == serializable_bprop_prim_name;
                     });
}

std::string GetBpropHash() {
  static std::string bprop_hash = std::string();
  if (bprop_hash.empty()) {
    auto bprop_dir = GetBpropDir();
    auto realpath = FileUtils::GetRealPath(common::SafeCStr(bprop_dir));
    if (!realpath.has_value()) {
      MS_LOG(EXCEPTION) << "Get real path of bprop dir failed. path=" << bprop_dir;
    }
    bprop_hash = system::sha256::GetHashFromDir(realpath.value());
  }
  return bprop_hash;
}

FuncGraphPtr ImportBpropFromMindIR(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  std::string bprop_dir = GetBpropDir();
  auto bprop_mindir_path = bprop_dir + kBpropMindIRDir;
  std::optional<std::string> bprop_mindir_realpath =
    FileUtils::GetRealPath(common::SafeCStr(bprop_mindir_path + prim->name() + kBpropMindIRSuffix));
  bool bprop_cache_file_exists = bprop_mindir_realpath.has_value() && Common::FileExists(bprop_mindir_realpath.value());
  if (!bprop_cache_file_exists) {
    return nullptr;
  }
  auto bprop_fg = LoadMindIR(bprop_mindir_realpath.value());
  if (bprop_fg != nullptr && bprop_fg->bprop_hash() != GetBpropHash()) {
    MS_LOG(EXCEPTION) << "The bprop mindir files are not up to date. Please run the " << bprop_mindir_path
                      << "generate_mindir.py to generate new mindir files.\n"
                      << "bprop_fg hash: " << bprop_fg->bprop_hash() << "\n"
                      << "bprop hash: " << GetBpropHash();
  }
  return bprop_fg;
}

void ExportBpropToMindIR(const PrimitivePtr &prim, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(prim);
  std::string bprop_dir = GetBpropDir();
  func_graph->set_bprop_hash(GetBpropHash());
  auto bprop_mindir_path = bprop_dir + kBpropMindIRDir;
  std::optional<std::string> bprop_mindir_realpath =
    Common::CreatePrefixPath(bprop_mindir_path + prim->name() + kBpropMindIRSuffix, true);
  if (!bprop_mindir_realpath.has_value()) {
    MS_LOG(ERROR) << "Failed to get the realpath of bprop mindir: " << bprop_mindir_path << prim->name()
                  << kBpropMindIRSuffix;
    return;
  }
  std::ofstream fout(bprop_mindir_realpath.value());
  mind_ir::ModelProto fg_model = GetBinaryProto(func_graph, false);
  if (!fg_model.SerializeToOstream(&fout)) {
    MS_LOG(WARNING) << "Failed to cache the bprop of op \"" << prim->name() << "\" to file \""
                    << bprop_mindir_realpath.value() << "\".";
  }
  fout.close();
  ChangeFileMode(bprop_mindir_realpath.value(), S_IRUSR | S_IWUSR);
}

AnfNodePtr GetPythonOps(const FuncGraphPtr &fg, const AnfNodePtr &origin_node, const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(origin_node);
  MS_EXCEPTION_IF_NULL(prim);
  // DoSignaturePrimitive to the pair of primitive name and module name.
  static std::unordered_map<std::string, std::pair<std::string, std::string>> python_ops{
    {"S-Prim-zeros_like_leaf", {"zeros_like", ""}},
    {"S-Prim-getitem", {"getitem", "mindspore.ops.composite.multitype_ops.getitem_impl"}}};
  auto iter = python_ops.find(prim->name());
  if (iter == python_ops.end()) {
    return nullptr;
  }
  ValuePtr python_ops_value;
  if (!iter->second.second.empty()) {
    python_ops_value = prim::GetPythonOps(iter->second.first, iter->second.second);
  } else {
    python_ops_value = prim::GetPythonOps(iter->second.first);
  }
  auto origin_cnode = origin_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_cnode);
  auto &origin_inputs = origin_cnode->inputs();
  std::vector<AnfNodePtr> new_inputs{NewValueNode(python_ops_value)};
  (void)std::copy(origin_inputs.begin() + 1, origin_inputs.end(), std::back_inserter(new_inputs));
  return fg->NewCNode(new_inputs);
}

// Replace the nodes whose python obj of primitive is needed in the renormalize process,
// with the new created python ops, such as zeros_like.
void ReplacePythonOps(const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(fg->get_return());
  for (const auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    for (size_t i = 0; i < cnode->size(); ++i) {
      auto prim = GetCNodePrimitive(cnode->input(i));
      if (prim == nullptr) {
        continue;
      }
      auto new_input = GetPythonOps(fg, cnode->input(i), prim);
      if (new_input == nullptr) {
        continue;
      }
      cnode->set_input(i, new_input);
    }
  }
}
#endif
}  // namespace

#ifndef _WIN32
// Given a python primitive, export a mindir file from the bprop defined in python.
void KPrim::ExportBpropMindir(const py::object &obj) {
  auto prim_adapter = obj.cast<PrimitivePyAdapterPtr>();
  if (prim_adapter == nullptr) {
    MS_LOG(EXCEPTION) << "The python obj to be exported to bprop mindir should be a Primitive, but it is "
                      << py::str(obj);
  }
  auto prim = prim_adapter->attached_primitive();
  if (prim == nullptr) {
    prim = std::make_shared<PrimitivePy>(obj, prim_adapter);
    prim_adapter->set_attached_primitive(prim);
  }

  // Get the bprop function from python.
  py::function fn = prim->cast<PrimitivePyPtr>()->GetBpropFunction();
  if (py::isinstance<py::none>(fn)) {
    fn = GetBpropFunction(prim->name());
  }
  if (!fn || py::isinstance<py::none>(fn)) {
    MS_LOG(EXCEPTION) << "Fail to find bprop function for " << prim->name() << ".";
  }
  auto func_graph = parse::ParsePythonCode(fn);
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Fail to parse bprop function for " << prim->name() << ".";
  }
  auto res = std::make_shared<pipeline::Resource>();
  (void)parse::ResolveFuncGraph(func_graph, res);
  ExportBpropToMindIR(prim, func_graph);
}
#endif

FuncGraphPtr KPrim::GetBprop(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &resources) {
  // Set a child scope named "grad'PrimitiveName'" for the bprop function,
  // and add "Gradients" to the front.
  static const std::string gradients_scope = "Gradients/";
  static const std::string grad_op_child_scope_prefix = "/grad";
  MS_EXCEPTION_IF_NULL(prim);
  auto scope = std::make_shared<Scope>(gradients_scope + ScopeManager::GetInstance().GetCurrentScope()->name() +
                                       grad_op_child_scope_prefix + prim->name());
  ScopeGuard scope_guard(scope);

  // Firstly we get bprop from mindir. If failed, parse the python function registered.
  FuncGraphPtr func_graph = nullptr;
#ifndef _WIN32
  bool serializable = IsSerializableBprop(prim->name());
  if (serializable) {
    func_graph = ImportBpropFromMindIR(prim);
    if (func_graph != nullptr) {
      ReplacePythonOps(func_graph);
      return func_graph;
    }
  }
#endif
  py::function fn;
  if (prim->is_base()) {
    fn = GetBpropFunction(prim->name());
  } else {
    fn = prim->cast<PrimitivePyPtr>()->GetBpropFunction();
    if (py::isinstance<py::none>(fn)) {
      fn = GetBpropFunction(prim->name());
    }
  }
  if (!fn || py::isinstance<py::none>(fn)) {
    MS_LOG(WARNING) << "Fail to find bprop function for " << prim->name() << ". fn: " << py::str(fn);
    return nullptr;
  }
  func_graph = parse::ParsePythonCode(fn);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Fail to parse bprop function for " << prim->name() << ".";
    return nullptr;
  }
  auto bprop_flag = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_BACKPROP);
  if (bprop_flag) {
    func_graph->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
  }
  pipeline::ResourceBasePtr res = (resources != nullptr) ? resources : std::make_shared<pipeline::Resource>();
  (void)parse::ResolveFuncGraph(func_graph, res);
#ifndef _WIN32
  // Check whether the bprop needs to be exported.
  if (serializable) {
    ExportBpropToMindIR(prim, func_graph);
  }
#endif
  return func_graph;
}

FuncGraphPtr KPrim::GetPossibleBprop(const PrimitivePtr &prim) {
  FuncGraphPtr bprop_fg = nullptr;
  auto iter = bprop_registry_.find(prim);
  if (iter != bprop_registry_.end()) {
    bprop_fg = iter->second;
  }

  if (bprop_fg == nullptr) {
    bprop_fg = GetBprop(prim);
    if (bprop_fg != nullptr) {
      // Set bprop_g graph cache
      bprop_registry_[prim] = bprop_fg;
    }
  }
  return bprop_fg;
}

FuncGraphPtr KPrim::GetFprop(const PrimitivePtr &prim) {
  static const std::string ad_module = "mindspore.ops._grad.grad_implementations";
  std::string func_name = "_fprop_" + prim->name();
  py::function fn = parse::python_adapter::GetPyFn(ad_module, func_name);
  auto func_graph = parse::ParsePythonCode(fn);
  MS_EXCEPTION_IF_NULL(func_graph);
  return BasicClone(func_graph);
}

MetaFuncGraphPtr KPrim::KMetaFuncGraph(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);

  auto iter = bprop_registry_meta_.find(prim);
  if (iter != bprop_registry_meta_.end()) {
    return iter->second;
  }

  if (prim->Hash() == prim::kPrimMakeTuple->Hash() && prim->name() == prim::kPrimMakeTuple->name()) {
    MetaFuncGraphPtr meta = std::make_shared<prim::MakeTupleGradient>("make_tuple_gradient");
    bprop_registry_meta_[prim::kPrimMakeTuple] = meta;
    return meta;
  }

  if (prim->Hash() == prim::kPrimMakeList->Hash() && prim->name() == prim::kPrimMakeList->name()) {
    MetaFuncGraphPtr meta = std::make_shared<prim::MakeListGradient>("make_list_gradient");
    bprop_registry_meta_[prim::kPrimMakeList] = meta;
    return meta;
  }

  MS_LOG(EXCEPTION) << "Fail to find bprop function for " << prim->name() << ".";
}

static void AppendMonadOutput(const FuncGraphPtr &bprop_fg, const AnfNodePtr &monad) {
  const auto &output = bprop_fg->output();
  MS_EXCEPTION_IF_NULL(output);
  auto output_cnode = output->cast<CNodePtr>();
  if (output_cnode != nullptr) {
    // If output_cnode has the form like (make_tuple, x, y).
    output_cnode->add_input(monad);
    return;
  }
  // If output is an empty tuple, create a (make_tuple, monad) as the new output.
  auto make_tuple = NewValueNode(prim::kPrimMakeTuple);
  output_cnode = bprop_fg->NewCNode({make_tuple, monad});
  bprop_fg->set_output(output_cnode);
}

// Append U or/and IO monad to output of Bprop funcgraph.
static void AdjustForAutoMonad(const PrimitivePtr &prim, const FuncGraphPtr &bprop_fg) {
  auto effect_info = GetPrimEffectInfo(prim);
  if (effect_info.memory) {
    MS_LOG(DEBUG) << "Append U monad for Bprop FuncGraph of Primitive " << prim->ToString();
    auto u = NewValueNode(kUMonad);
    u->set_abstract(kUMonad->ToAbstract());
    AppendMonadOutput(bprop_fg, u);
  }
  if (effect_info.io) {
    MS_LOG(DEBUG) << "Append IO monad for Bprop FuncGraph of Primitive " << prim->ToString();
    auto io = NewValueNode(kIOMonad);
    io->set_abstract(kIOMonad->ToAbstract());
    AppendMonadOutput(bprop_fg, io);
  }
}

FuncGraphPtr KPrim::GetBprop(const CNodePtr &cnode, const ValueNodePtr &value_node,
                             const pipeline::ResourceBasePtr &resources, const PrimitivePtr &prim) {
  FuncGraphPtr bprop_fg = nullptr;
  if (prim->Hash() == prim::kPrimHookBackward->Hash() && prim->name() == prim::kPrimHookBackward->name()) {
    if (MsContext::GetInstance()->get_param<int>(MsCtxParam::MS_CTX_EXECUTION_MODE) == kGraphMode) {
      MS_LOG(EXCEPTION)
        << "The Primitive 'HookBackward' is not supported in graph mode, which is only supported in pynative mode.\n"
        << trace::GetDebugInfo(cnode->debug_info());
    }
    bprop_fg = BpropCut(value_node, resources);
  } else {
    auto iter = bprop_registry_.find(prim);
    if (iter != bprop_registry_.end()) {
      bprop_fg = iter->second;
    }

    if (bprop_fg == nullptr) {
      bprop_fg = GetBprop(prim, resources);
      if (bprop_fg != nullptr) {
        // Set bprop_g graph cache
        bprop_registry_[prim] = bprop_fg;
      } else {
        bprop_fg = FakeBprop(value_node, resources);
      }
    }
  }
  return bprop_fg;
}

FuncGraphPtr KPrim::KPrimitive(const CNodePtr &cnode, const ValueNodePtr &value_node,
                               const pipeline::ResourceBasePtr &resources) {
  if (!IsValueNode<Primitive>(value_node)) {
    MS_LOG(EXCEPTION) << "Primitive node is not valid.";
  }

  auto prim = GetValueNode<PrimitivePtr>(value_node);
  if (prim->Hash() == prim::kPrimSwitchLayer->Hash() && prim->name() == prim::kPrimSwitchLayer->name()) {
    auto fprop = GetFprop(prim);
    fprop->transforms().emplace("primal", FuncGraphTransform(prim::kPrimSwitchLayer));
    return fprop;
  } else if (prim->Hash() == prim::kPrimMakeTuple->Hash() && prim->name() == prim::kPrimMakeTuple->name()) {
    return nullptr;
  } else if (prim->Hash() == prim::kPrimMakeList->Hash() && prim->name() == prim::kPrimMakeList->name()) {
    return nullptr;
  }

  FuncGraphPtr bprop_fg = GetBprop(cnode, value_node, resources, prim);
  AdjustForAutoMonad(prim, bprop_fg);
  std::unordered_map<std::string, ValuePtr> primal_attrs;
  std::vector<NodeDebugInfoPtr> primal_debug_infos;
  if (resources != nullptr) {
    auto manager = resources->manager();
    auto &users = manager->node_users()[value_node];
    for (auto user_iter = users.begin(); user_iter != users.end(); ++user_iter) {
      primal_debug_infos.push_back(user_iter->first->debug_info());
    }
  }
  if (cnode != nullptr) {
    primal_attrs = cnode->primal_attrs();
    const auto forward_node_primal_attr = prim->name() + "_" + cnode->UniqueId();
    primal_attrs[kPrimalAttrForwardNodeName] = MakeValue(forward_node_primal_attr);
  }
  auto expanded_fg = BpropToK(prim, bprop_fg, nullptr, cnode, primal_attrs, primal_debug_infos);
  if (expanded_fg == nullptr) {
    MS_LOG(EXCEPTION) << "Failed convert " << prim->name()
                      << " prim bprop function to J expanded func graph. NodeInfo: "
                      << trace::GetDebugInfo(bprop_fg->debug_info());
  }
  if (lift_fv_before_grad && IsPrimitiveEquals(prim, prim::kPrimSwitch)) {
    // Inline fprop_switch before renormalize;
    expanded_fg->set_flag(FUNC_GRAPH_FLAG_FORCE_INLINE, true);
    MS_LOG(DEBUG) << "set force_inline for fg: " << expanded_fg->ToString();
  }

  return expanded_fg;
}

AnfNodePtr KPrim::BuildOutput(const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg) {
  // current_primal_fg may have extra parameters like u_monad, io_monad
  std::vector<AnfNodePtr> extra_args;
  // caller had checked size() - 2 is greater than 0.
  auto bprop_fg_param_size = bprop_fg->parameters().size() - 2;
  if (current_primal_fg != nullptr && bprop_fg_param_size < current_primal_fg->parameters().size()) {
    auto current_primal_fg_param_size = current_primal_fg->parameters().size();
    MS_LOG(DEBUG) << "Current Primal FuncGraph may have extra parameters(U or IO monad) which bprop don't define, so "
                     "Insert it. Extra parameters size: "
                  << current_primal_fg_param_size - bprop_fg_param_size;
    for (auto i = bprop_fg_param_size; i < current_primal_fg_param_size; ++i) {
      const auto &primal_node = current_primal_fg->parameters()[i];
      AnfNodePtr extra_node;
      // Simplify zeros_like(primal_node) to U or IO, so extra_node in bprop_fg will not refer to primal_node
      // as a free variable of primal_graph.
      // Notes: if the implementation of zeros_like changes, here too.
      if (HasAbstractUMonad(primal_node)) {
        extra_node = NewValueNode(kUMonad);
      } else if (HasAbstractIOMonad(primal_node)) {
        extra_node = NewValueNode(kIOMonad);
      } else {
        MS_EXCEPTION(TypeError)
          << "The params of function 'bprop' of Primitive or Cell requires the forward inputs as well "
             "as the 'out' and 'dout'.\n"
          << trace::GetDebugInfo(bprop_fg->debug_info());
      }
      extra_args.push_back(extra_node);
      MS_LOG(DEBUG) << "Insert to bprop_fg for node: " << primal_node->DebugString();
    }
  }
  // bprop_fg has been checked in caller
  if (IsPrimitiveCNode(bprop_fg->output(), prim::kPrimMakeTuple)) {
    // Set bprop output as (env, dx, dy, dz, ...)
    auto cbprop = bprop_fg->output()->cast<CNodePtr>();
    auto &inputs = cbprop->inputs();

    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(prim::kPrimMakeTuple));
    args.push_back(NewValueNode(newenv));
    (void)args.insert(args.end(), inputs.begin() + 1, inputs.end());
    if (!extra_args.empty()) {
      args.insert(args.end(), extra_args.cbegin(), extra_args.cend());
    }
    return NewCNode(args, bprop_fg);
  }

  // Set bprop output as (env, dx)
  std::string model_name("mindspore.ops.composite.multitype_ops.add_impl");
  std::string python_ops("_tuple_add");
  auto tuple_env = NewCNode({NewValueNode(prim::kPrimMakeTuple), NewValueNode(newenv)}, bprop_fg);
  auto tuple_add_ops = NewValueNode(prim::GetPythonOps(python_ops, model_name));
  if (!extra_args.empty()) {
    extra_args.insert(extra_args.begin(), NewValueNode(prim::kPrimMakeTuple));
    auto extra_tuple = NewCNode(extra_args, bprop_fg);
    auto old_output_extra = NewCNode({tuple_add_ops, bprop_fg->output(), extra_tuple}, bprop_fg);
    return NewCNode({tuple_add_ops, tuple_env, old_output_extra}, bprop_fg);
  }

  return NewCNode({tuple_add_ops, tuple_env, bprop_fg->output()}, bprop_fg);
}

static void TransformNormalArgs(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg, const FuncGraphPtr &outer,
                                std::vector<AnfNodePtr> *const transf_args) {
  // bprop_fg has been checked in caller
  // transform except the last 2 parameters: out, dout.
  const size_t last_parameter_sizes = 2;
  auto bprop_fg_param_size = bprop_fg->parameters().size() - last_parameter_sizes;
  for (size_t i = 0; i < bprop_fg_param_size; ++i) {
    auto p = bprop_fg->parameters()[i];
    MS_EXCEPTION_IF_NULL(p);

    TraceGuard trace_guard(std::make_shared<TraceGradFprop>(p->debug_info()));
    auto transf_p = outer->add_parameter();

    (void)mng->Replace(p, transf_p);
    transf_args->push_back(transf_p);
  }
}
void KPrim::TransformArgsForPrimitive(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg,
                                      const PrimitivePtr &primitive, const FuncGraphPtr &outer,
                                      std::vector<AnfNodePtr> *const transf_args) {
  MS_EXCEPTION_IF_NULL(mng);
  TransformNormalArgs(mng, bprop_fg, outer, transf_args);
  // Fprop_fg for Primitive with side effect should append extra U or IO monad parameter.
  auto effect_info = GetPrimEffectInfo(primitive);
  if (effect_info.memory) {
    MS_LOG(DEBUG) << "Append U monad to Fprop FuncGraph for Primitive " << primitive->ToString();
    auto transf_p = outer->add_parameter();
    transf_args->push_back(transf_p);
  }
  if (effect_info.io) {
    MS_LOG(DEBUG) << "Append IO monad to Fprop FuncGraph for Primitive " << primitive->ToString();
    auto transf_p = outer->add_parameter();
    transf_args->push_back(transf_p);
  }
}

template <typename T>
void KPrim::TransformArgsForFuncGraph(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg,
                                      const T &current_primal_fg, const FuncGraphPtr &outer,
                                      std::vector<AnfNodePtr> *const transf_args) {
  MS_EXCEPTION_IF_NULL(mng);
  TransformNormalArgs(mng, bprop_fg, outer, transf_args);
  constexpr size_t need_filter_size = 2;
  auto bprop_fg_param_size = bprop_fg->parameters().size() - need_filter_size;
  // current_primal_fg may have extra parameters after AutoMonad
  const auto &current_primal_fg_params = current_primal_fg->parameters();
  if (bprop_fg_param_size < current_primal_fg_params.size()) {
    for (auto i = bprop_fg_param_size; i < current_primal_fg_params.size(); ++i) {
      auto p = current_primal_fg_params[i];
      MS_EXCEPTION_IF_NULL(p);
      // extra parameters should be Monad.
      if (!HasAbstractMonad(p)) {
        continue;
      }
      MS_LOG(DEBUG) << "Function " << current_primal_fg->ToString()
                    << ", has extra monad parameter: " << p->DebugString()
                    << ", abstract: " << p->abstract()->ToString();

      TraceGuard trace_guard(std::make_shared<TraceGradFprop>(p->debug_info()));
      auto transf_p = outer->add_parameter();
      // See also Notes on extra_node of BuildOutput.
      // Notes: No need to replace p with transf_p as the only use of p is here.
      // If extra_node in bprop_fg use p as free variable, a replacement of p is required here.
      // This replacement will make the usage of p in current_primal_fg got replaced with transf_p
      // of outer. outer will be released after it is being cloned to fprop_fg, so the func_graph_
      // in transf_p will be nullptr.
      // So the RULE is DONT tamper the current_primal_fg;
      transf_args->push_back(transf_p);
    }
  }
  if (transf_args->size() != current_primal_fg_params.size()) {
    MS_EXCEPTION(TypeError) << "Function " << current_primal_fg->ToString()
                            << ", The number of parameter of this primal function is "
                            << current_primal_fg_params.size() << ", but the number of parameters of bprop is "
                            << bprop_fg_param_size;
  }
}

void KPrim::CheckBprop(const FuncGraphPtr &bprop_fg, const string &prim_to_check) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool check_bprop_flag = context->get_param<bool>(MS_CTX_CHECK_BPROP_FLAG);
  // Skip checking if check_bprop not set
  if (!check_bprop_flag) {
    return;
  }

  // bprop_fg has been checked in caller
  auto check_bprop_class = prim::GetPythonOps("CheckBprop", "mindspore.ops.operations.other_ops");
  MS_EXCEPTION_IF_NULL(check_bprop_class);
  auto check_bprop =
    bprop_fg->NewCNode({NewValueNode(check_bprop_class), NewValueNode(std::make_shared<StringImm>(prim_to_check))});

  std::vector<AnfNodePtr> inputs;
  inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  constexpr int primitive_size = 1;
  constexpr int brprop_offset_size = 2;
  (void)inputs.insert(inputs.begin() + primitive_size, bprop_fg->parameters().begin(),
                      bprop_fg->parameters().end() - brprop_offset_size);
  AnfNodePtr params = bprop_fg->NewCNode(inputs);

  inputs.clear();
  inputs.push_back(check_bprop);
  inputs.push_back(bprop_fg->output());
  inputs.push_back(params);
  AnfNodePtr bprop_out = bprop_fg->NewCNode(inputs);
  bprop_fg->set_output(bprop_out);
}

FuncGraphPtr KPrim::KUserDefinedCellBprop(const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg) {
  MS_EXCEPTION_IF_NULL(bprop_fg);
  // primal_fg is FuncGraph just after convert. Refer ConvertCellObjToFuncGraph.
  // current_primal_fg is specalized and AutoMoaded primal_fg;
  auto primal_fg = bprop_fg->transforms().find("primal")->second.func_graph();
  auto expanded_fg = BpropToK(primal_fg, bprop_fg, current_primal_fg, nullptr, {}, {});
  if (expanded_fg == nullptr) {
    MS_LOG(EXCEPTION) << "Failed convert " << primal_fg->ToString()
                      << " Cell bprop function to K expanded func graph. NodeInfo: "
                      << trace::GetDebugInfo(primal_fg->debug_info());
  }
  return expanded_fg;
}

FuncGraphPtr KPrim::BpropCut(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) {
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto &node_users = resources->manager()->node_users();

  auto &users = node_users[value_node];
  auto cnode = std::find_if(users.begin(), users.end(), [&prim](const std::pair<AnfNodePtr, int64_t> &user) -> bool {
    return IsPrimitiveCNode(user.first, prim);
  });
  if (cnode == users.end()) {
    MS_LOG(EXCEPTION) << "Fail to find cnode.";
  }
  auto inputs_num = cnode->first->cast<CNodePtr>()->size() - 1;

  auto func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;
  auto bprop_cut = std::make_shared<PrimitivePy>("bprop_cut");
  bprop_cut->CopyHookFunction(prim);

  auto cell_id = GetValue<std::string>(prim->GetAttr("cell_id"));
  if (cell_id != "") {
    (void)bprop_cut->AddAttr("cell_hook", MakeValue(true));
    (void)bprop_cut->AddAttr("cell_id", MakeValue(cell_id));
  }

  outputs.push_back(NewValueNode(bprop_cut));
  for (size_t i = 0; i < inputs_num; ++i) {
    auto param = func_graph->add_parameter();
    outputs.push_back(param);
  }
  auto p1 = func_graph->add_parameter();
  auto p2 = func_graph->add_parameter();
  outputs.push_back(p1);
  outputs.push_back(p2);

  func_graph->set_output(func_graph->NewCNode(outputs));
  return func_graph;
}

FuncGraphPtr KPrim::FakeBprop(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) {
  auto prim = value_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);
  auto &node_users = resources->manager()->node_users();

  auto &users = node_users[value_node];
  auto cnode = std::find_if(users.begin(), users.end(), [&prim](const std::pair<AnfNodePtr, int64_t> &user) -> bool {
    return IsPrimitiveCNode(user.first, prim);
  });
  if (cnode == users.end()) {
    MS_LOG(EXCEPTION) << "Fail to find user for " << prim->ToString();
  }
  auto inputs_num = cnode->first->cast<CNodePtr>()->inputs().size() - 1;
  auto effect_info = GetPrimEffectInfo(prim);
  // Don't add U or IO monad parameters as it will be added later.
  size_t monad_params_size = 0;
  if (effect_info.memory) {
    monad_params_size++;
  }
  if (effect_info.io) {
    monad_params_size++;
  }
  if (inputs_num < monad_params_size) {
    MS_LOG(EXCEPTION) << "Arguments number should be greater than or equal to " << monad_params_size
                      << ", but the CNode is: " << cnode->first->DebugString();
  }
  inputs_num -= monad_params_size;

  auto func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;
  outputs.push_back(NewValueNode(prim::kPrimMakeTuple));

  auto fake_bprop = std::make_shared<Primitive>("fake_bprop");
  (void)fake_bprop->AddAttr("info", MakeValue("Primitive " + prim->name() + "'s bprop not defined."));

  for (size_t i = 0; i < inputs_num; ++i) {
    // Mock params for inputs
    auto param = func_graph->add_parameter();
    // Mock derivatives for each inputs
    outputs.push_back(func_graph->NewCNode({NewValueNode(fake_bprop), param}));
  }
  // mock params for out and dout
  (void)func_graph->add_parameter();
  (void)func_graph->add_parameter();
  func_graph->set_output(func_graph->NewCNode(outputs));
  return func_graph;
}
}  // namespace ad
}  // namespace mindspore
