/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/ad/bprop_utils.h"

#include <string>
#include <regex>
#include <utility>
#include <algorithm>
#include <vector>
#include <memory>
#include "include/common/utils/primitive_utils.h"
#include "include/common/debug/common.h"
#include "utils/file_utils.h"
#include "utils/system/sha256.h"
#include "mindspore/core/load_mindir/load_model.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/pynative/grad/bprop_expander/bprop.h"
#include "pipeline/pynative/grad/bprop_expander/bprop_irbuilder.h"
#include "frontend/optimizer/expander.h"
#include "include/common/debug/dump_proto.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/operator/graph_bprop/bprop_meta_func_graph.h"

namespace mindspore {
namespace ad {
namespace {
constexpr char kBpropMindIRDir[] = "/../bprop_mindir/";
constexpr char kBpropMindIRSuffix[] = "_bprop.mindir";
#ifndef _WIN32
std::string GetBpropDir() {
  static std::string bprop_dir;
  if (bprop_dir.empty()) {
    py::module mod = py::module::import("mindspore.ops._grad");
    auto grad_file_path = mod.attr("__file__").cast<std::string>();
    bprop_dir = grad_file_path.substr(0, grad_file_path.find_last_of('/'));
  }
  return bprop_dir;
}

bool IsSerializableBprop(const std::string &prim_name) {
  static const std::string bprop_mindir_path = GetBpropDir() + kBpropMindIRDir;
  std::string bprop_mindir_realpath = bprop_mindir_path + prim_name + kBpropMindIRSuffix;
  return Common::FileExists(bprop_mindir_realpath);
}

std::string GetBpropString(const std::string &bprop_file_path, const std::string &prim_name) {
  std::ifstream file(bprop_file_path);
  if (!file.is_open()) {
    MS_LOG(ERROR) << "Failed to open file: " << bprop_file_path;
    return "";
  }
  std::string line;
  std::string bprop_str;
  std::regex pattern(R"(@bprop(_getter)?s\.register\(.*[."]?)" + prim_name + R"((")?\))");
  bool match_register = false;
  bool match_def = false;
  while (getline(file, line)) {
    if (match_def) {
      if (line.find("    ") != 0 && !line.empty()) {
        break;
      }
      bprop_str += line + "/n";
    }
    if (!match_register && std::regex_match(line, pattern)) {
      match_register = true;
    }
    if (match_register && !match_def && line.find("def") == 0) {
      match_def = true;
    }
  }
  return bprop_str;
}

std::pair<std::string, std::string> GetBpropHashAndFilePath(const py::function &fn, const std::string &prim_name) {
  // Get the file where the bprop function is defined.
  auto filename = fn.attr("__code__").attr("co_filename").cast<std::string>();
  auto realpath = FileUtils::GetRealPath(common::SafeCStr(filename));
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Failed to get the realpath of file: " << filename;
    return std::make_pair("", "");
  }
  // Get the hash of the function.
  auto bprop_str = GetBpropString(realpath.value(), prim_name);
  MS_LOG(DEBUG) << "bprop string: " << bprop_str;
  // Get the relative path of the function.
  auto filepath = realpath.value();
  static std::string bprop_dir = GetBpropDir();
  if (filepath.find(bprop_dir) == 0) {
    filepath = filepath.substr(bprop_dir.length());
  } else {
    MS_LOG(ERROR) << "The realpath of the bprop function do not contain the bprop dir, realpath: " << filepath
                  << ", bprop dir: " << bprop_dir;
    return std::make_pair("", "");
  }
  return std::make_pair(system::sha256::GetHashFromString(bprop_str), filepath);
}

bool CheckBpropUpToDate(const std::string &prim_name, const std::string &current_hash) {
  static const std::string bprop_mindir_path = GetBpropDir() + kBpropMindIRDir;
  std::string bprop_mindir_realpath = bprop_mindir_path + prim_name + kBpropMindIRSuffix;
  if (!Common::FileExists(bprop_mindir_realpath)) {
    return false;
  }
  MindIRLoader mindir_loader;
  auto bprop_fg = mindir_loader.LoadMindIR(bprop_mindir_realpath);
  if (bprop_fg == nullptr) {
    MS_LOG(WARNING) << "Failed to load the bprop mindir " << bprop_mindir_realpath;
    return false;
  }
  return bprop_fg->bprop_hash() == current_hash;
}

bool ExportBpropToMindIR(const std::string &prim_name, const FuncGraphPtr &func_graph) {
  static const auto bprop_mindir_dir = GetBpropDir() + kBpropMindIRDir;
  std::string bprop_mindir_path = bprop_mindir_dir + prim_name + kBpropMindIRSuffix;
  return DumpBinaryProto(func_graph, bprop_mindir_path);
}

bool CheckBpropHash(const std::string &prim_name, const std::string &hash_in_mindir,
                    const std::string &bprop_filepath) {
  auto real_path = GetBpropDir() + bprop_filepath;
  auto bprop_hash = system::sha256::GetHashFromString(GetBpropString(real_path, prim_name));
  if (hash_in_mindir == bprop_hash) {
    return true;
  }
  std::string bprop_dir = GetBpropDir();
  auto bprop_mindir_path = bprop_dir + kBpropMindIRDir;
  MS_LOG(ERROR) << "The bprop mindir files of " << prim_name << " is not up to date. Please run the "
                << bprop_mindir_path << "generate_mindir.py to generate new mindir files.\n"
                << "the hash of bprop function: " << bprop_hash << "\n"
                << "the hash in mindir: " << hash_in_mindir;

  return false;
}

FuncGraphPtr ImportBpropFromMindIR(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  static auto bprop_mindir_path = GetBpropDir() + kBpropMindIRDir;
  std::optional<std::string> bprop_mindir_realpath =
    FileUtils::GetRealPath(common::SafeCStr(bprop_mindir_path + prim->name() + kBpropMindIRSuffix));
  bool bprop_cache_file_exists = bprop_mindir_realpath.has_value() && Common::FileExists(bprop_mindir_realpath.value());
  if (!bprop_cache_file_exists) {
    return nullptr;
  }
  MindIRLoader mindir_loader;
  auto bprop_fg = mindir_loader.LoadMindIR(bprop_mindir_realpath.value());
  if (bprop_fg == nullptr) {
    MS_LOG(WARNING) << "Failed to load the bprop mindir " << bprop_mindir_realpath.value();
    return nullptr;
  }
  if (!CheckBpropHash(prim->name(), bprop_fg->bprop_hash(), bprop_fg->bprop_filepath())) {
    MS_LOG(EXCEPTION) << "The bprop mindir files are not up to date. The name of prim is: " << prim->name() << ".";
  }
  return bprop_fg;
}

void EliminateParameterSelf(const FuncGraphPtr &func_graph, const pipeline::ResourceBasePtr &resources,
                            const PrimitivePtr &prim) {
  auto parameters = func_graph->parameters();
  if (parameters.empty()) {
    return;
  }
  auto para0 = parameters[0]->cast_ptr<Parameter>();
  MS_EXCEPTION_IF_NULL(para0);
  if (para0->name() != "self") {
    return;
  }
  auto mng = resources->manager();
  MS_EXCEPTION_IF_NULL(mng);
  mng->AddFuncGraph(func_graph);
  mng->Replace(parameters[0], NewValueNode(prim));
  std::vector<AnfNodePtr> new_parameters;
  (void)std::copy(parameters.begin() + 1, parameters.end(), std::back_inserter(new_parameters));
  func_graph->set_parameters(new_parameters);
}

// Reload the python obj or other types which can't be stored in mindir.
void OptimizeBpropFromMindIR(const FuncGraphPtr &fg, const pipeline::ResourceBasePtr &resources,
                             const PrimitivePtr &prim) {
  auto res = (resources != nullptr) ? resources : std::make_shared<pipeline::Resource>();
  EliminateParameterSelf(fg, res, prim);
  opt::irpass::BpropMindIRPassLib irpass;

  std::vector<opt::SubstitutionPtr> opt_list{irpass.reslove_primitive_attr_, irpass.get_constexpr_ops_,
                                             irpass.get_class_type_,         irpass.get_meta_fg_,
                                             irpass.get_primal_attr_,        irpass.get_sub_func_graph_};
  opt::OptPassGroupMap map({
    {"bprop_mindir_opt", opt::OptPassConfig(opt_list)},
  });
  opt::OptimizerPtr bprop_mindir_opt = opt::Optimizer::MakeOptimizer("bprop_mindir_opt", res, map, false, false, false);
  (void)bprop_mindir_opt->step(fg, false);
}

FuncGraphPtr LiftParameter(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  if (!IsValueNode<FuncGraph>(func_graph->output())) {
    return func_graph;
  }
  auto bprop_fg = GetValueNode<FuncGraphPtr>(func_graph->output());
  MS_EXCEPTION_IF_NULL(bprop_fg);
  for (auto &p : bprop_fg->parameters()) {
    TraceGuard trace_guard(std::make_shared<TraceCopy>(p->debug_info()));
    auto new_p = func_graph->add_parameter();
    mng->Replace(p, new_p);
  }
  bprop_fg->set_parameters({});
  auto call_bprop = func_graph->NewCNode({func_graph->output()});
  func_graph->set_output(call_bprop);
  return func_graph;
}

FuncGraphPtr RemovePyObj(const FuncGraphPtr &func_graph, const pipeline::ResourceBasePtr &res) {
  opt::irpass::BpropMindIRPassLib irpass;
  opt::OptPassGroupMap map({
    {"remove_resolve", opt::OptPassConfig({irpass.resolve_node_resolve_})},
    {"remove_class_type", opt::OptPassConfig({irpass.class_type_resolve_})},
    {"resolve_do_signature_prim", opt::OptPassConfig({irpass.do_signature_resolve_})},
  });
  opt::OptimizerPtr export_mindir_opt =
    opt::Optimizer::MakeOptimizer("export_mindir_opt", res, map, false, false, false);
  (void)export_mindir_opt->step(func_graph, false);
  return func_graph;
}
#endif
}  // namespace

#ifndef _WIN32
void ExportBpropToMindir(const py::object &obj, bool force_update = false) {
  if (!py::isinstance<py::str>(obj)) {
    MS_LOG(EXCEPTION) << "The python obj " << py::str(obj) << " to be exported to mindir should be a string";
  }
  auto prim_name = obj.cast<std::string>();
  // Get the bprop function from python.
  py::function fn = GetBpropFunctionByObj(obj, true);
  if (!fn || py::isinstance<py::none>(fn)) {
    MS_LOG(EXCEPTION) << "Fail to find bprop function for " << prim_name << ".";
  }
  auto bprop_hash_and_filepath = GetBpropHashAndFilePath(fn, prim_name);
  if (bprop_hash_and_filepath.first.empty()) {
    MS_LOG(EXCEPTION) << "Fail to get the function hash of bprop for " << prim_name;
  }
  if (bprop_hash_and_filepath.second.empty()) {
    MS_LOG(EXCEPTION) << "Fail to get the file path of bprop for " << prim_name;
  }
  // If the bprop file hash has not changed, we don't need to export a new mindir.
  if (!force_update && CheckBpropUpToDate(prim_name, bprop_hash_and_filepath.first)) {
    MS_LOG(WARNING) << "The hash of bprop function of primitive " << prim_name
                    << " is not changed, we will not export a new mindir.";
    return;
  }
  // Parse and resolve.
  auto func_graph = parse::ParsePythonCode(fn);
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Fail to parse bprop function for " << prim_name << ".";
  }
  auto res = std::make_shared<pipeline::Resource>();
  (void)parse::ResolveFuncGraph(func_graph, res);
  // Lift the parameters of the sub function to the top.
  func_graph = LiftParameter(func_graph);
  // For the mindir don't support to save py obj.
  func_graph = RemovePyObj(func_graph, res);
  func_graph->set_bprop_hash(bprop_hash_and_filepath.first);
  func_graph->set_bprop_filepath(bprop_hash_and_filepath.second);
  if (!ExportBpropToMindIR(prim_name, func_graph)) {
    MS_LOG(EXCEPTION) << "Failed to export the bprop mindir for " << prim_name;
  }
}

bool CheckMindir(const py::object &obj) {
  if (!py::isinstance<py::str>(obj)) {
    MS_LOG(EXCEPTION) << "The python obj " << py::str(obj) << " to be exported to mindir should be a string";
  }
  auto prim_name = obj.cast<std::string>();
  static auto bprop_mindir_path = GetBpropDir() + kBpropMindIRDir;
  std::optional<std::string> bprop_mindir_realpath =
    FileUtils::GetRealPath(common::SafeCStr(bprop_mindir_path + prim_name + kBpropMindIRSuffix));
  bool bprop_cache_file_exists = bprop_mindir_realpath.has_value() && Common::FileExists(bprop_mindir_realpath.value());
  if (!bprop_cache_file_exists) {
    MS_LOG(ERROR) << "There should be a bprop mindir file of " << prim_name;
    return false;
  }
  MindIRLoader mindir_loader;
  auto bprop_fg = mindir_loader.LoadMindIR(bprop_mindir_realpath.value());
  if (bprop_fg == nullptr) {
    MS_LOG(ERROR) << "Failed to load the bprop mindir " << bprop_mindir_realpath.value();
    return false;
  }
  if (!CheckBpropHash(prim_name, bprop_fg->bprop_hash(), bprop_fg->bprop_filepath())) {
    MS_LOG(ERROR) << "The bprop mindir files are not up to date.";
    return false;
  }
  return true;
}
#endif

FuncGraphPtr GetBprop(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &resources, const CNodePtr &cnode) {
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
  if (common::GetEnv("MS_DEV_GET_PYTHON_BPROP") != "1") {
    const auto &bprop_impl_map = graph_bprop::GetPrimitiveBpropImplMap();
    auto iter = bprop_impl_map.find(prim->name());
    if (iter != bprop_impl_map.end()) {
      std::vector<AnfNodePtr> node_lists = cnode->inputs();
      auto forward_inputs_size = cnode->inputs().size() - 1;
      for (size_t i = 1; i < node_lists.size(); i++) {
        auto input_i = node_lists[i];
        if (HasAbstractMonad(input_i)) {
          --forward_inputs_size;
        }
      }
      func_graph = iter->second(prim, forward_inputs_size);
      MS_EXCEPTION_IF_NULL(func_graph);
      func_graph->set_flag(mindspore::kFuncGraphFlagMetaFuncGraphBprop, true);
      if (GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_BACKPROP)) {
        func_graph->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
      }
      return func_graph;
    }
  }
#ifndef _WIN32
  if (IsSerializableBprop(prim->name())) {
    func_graph = ImportBpropFromMindIR(prim);
    if (func_graph != nullptr) {
      OptimizeBpropFromMindIR(func_graph, resources, prim);
      if (GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_BACKPROP)) {
        func_graph->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
      }
      return func_graph;
    }
  }
#endif
  py::function fn;
  if (prim->is_base()) {
    fn = GetBpropFunction(prim->name());
  } else {
    fn = prim->cast_ptr<PrimitivePy>()->GetBpropFunction();
    if (py::isinstance<py::none>(fn)) {
      fn = GetBpropFunction(prim->name());
    }
  }
  if (!fn || py::isinstance<py::none>(fn)) {
    MS_LOG(INFO) << "Fail to find bprop function for " << prim->name() << ". fn: " << py::str(fn);
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
  (void)parse::ResolveFuncGraph(func_graph, res, false);
  return func_graph;
}
}  // namespace ad
}  // namespace mindspore
