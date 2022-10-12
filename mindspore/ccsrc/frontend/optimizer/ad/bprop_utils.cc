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
#include "include/common/debug/dump_proto.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"

namespace mindspore {
namespace ad {
namespace {
constexpr char kBpropMindIRDir[] = "/../bprop_mindir/";
constexpr char kBpropMindIRSuffix[] = "_bprop.mindir";
constexpr char kBpropMindirModule[] = "mindspore.ops.bprop_mindir";
constexpr char kSerializableBpropOps[] = "serializable_bprop_ops";
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

bool BpropMindirDirExists() {
  auto bprop_mindir_dir = GetBpropDir() + kBpropMindIRDir;
  DIR *dir = opendir(bprop_mindir_dir.c_str());
  if (dir != nullptr) {
    if (closedir(dir) == -1) {
      MS_LOG(WARNING) << "The bprop mindir dir \"" << bprop_mindir_dir << "\" close failed!";
    }
    return true;
  }
  MS_LOG(ERROR) << "Open bprop mindir dir \"" << bprop_mindir_dir << "\" failed." << ErrnoToString(errno);
  return false;
}

// Get the serializable bprop list from the module mindspore.ops.bprop_mindir in python.
mindspore::HashSet<std::string> GetSerializableBpropList() {
  mindspore::HashSet<std::string> serializable_bprop_list;
  if (!BpropMindirDirExists()) {
    return serializable_bprop_list;
  }
  py::module mod = py::module::import(kBpropMindirModule);
  py::object serializable_bprop_ops_attr = mod.attr(kSerializableBpropOps);
  if (!py::isinstance<py::list>(serializable_bprop_ops_attr)) {
    MS_LOG(WARNING) << "Can not get the the serializable bprop ops list from python, it is not a python list.";
    return serializable_bprop_list;
  }

  auto ops_list = serializable_bprop_ops_attr.cast<py::list>();
  for (auto op : ops_list) {
    if (py::isinstance<py::str>(op)) {
      (void)serializable_bprop_list.insert(op.cast<std::string>());
      continue;
    }
    py::object prim_name = op.attr("__name__");
    if (!py::isinstance<py::str>(prim_name)) {
      MS_LOG(WARNING) << "The name of obj " << py::str(op) << " to be exported to mindir should be a string";
      continue;
    }
    (void)serializable_bprop_list.insert(prim_name.cast<std::string>());
  }
  return serializable_bprop_list;
}

bool IsSerializableBprop(const std::string &prim_name) {
  static mindspore::HashSet<std::string> serializable_bprop_list = GetSerializableBpropList();
  return std::any_of(serializable_bprop_list.begin(), serializable_bprop_list.end(),
                     [&prim_name](const std::string &serializable_bprop_prim_name) {
                       return prim_name == serializable_bprop_prim_name;
                     });
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

bool NeedExportBpropMindIR(const std::string &prim_name, const std::string &current_hash) {
  static const std::string bprop_mindir_path = GetBpropDir() + kBpropMindIRDir;
  std::optional<std::string> bprop_mindir_realpath =
    FileUtils::GetRealPath(common::SafeCStr(bprop_mindir_path + prim_name + kBpropMindIRSuffix));
  bool bprop_cache_file_exists = bprop_mindir_realpath.has_value() && Common::FileExists(bprop_mindir_realpath.value());
  if (!bprop_cache_file_exists) {
    return true;
  }
  MindIRLoader mindir_loader;
  auto bprop_fg = mindir_loader.LoadMindIR(bprop_mindir_realpath.value());
  if (bprop_fg == nullptr) {
    MS_LOG(WARNING) << "Failed to load the bprop mindir " << bprop_mindir_realpath.value();
    return true;
  }
  return bprop_fg->bprop_hash() != current_hash;
}

bool ExportBpropToMindIR(const std::string &prim_name, const FuncGraphPtr &func_graph) {
  static const auto bprop_mindir_dir = GetBpropDir() + kBpropMindIRDir;
  std::string bprop_mindir_path = bprop_mindir_dir + prim_name + kBpropMindIRSuffix;
  return DumpBinaryProto(func_graph, bprop_mindir_path);
}

bool CheckBpropHash(const PrimitivePtr &prim, const std::string &hash_in_mindir, const std::string &bprop_filepath) {
  auto real_path = GetBpropDir() + bprop_filepath;
  auto bprop_hash = system::sha256::GetHashFromString(GetBpropString(real_path, prim->name()));
  if (hash_in_mindir == bprop_hash) {
    return true;
  }
  std::string bprop_dir = GetBpropDir();
  auto bprop_mindir_path = bprop_dir + kBpropMindIRDir;
  MS_LOG(ERROR) << "The bprop mindir files of " << prim->name() << " is not up to date. Please run the "
                << bprop_mindir_path << "generate_mindir.py to generate new mindir files.\n"
                << "the hash of bprop function: " << bprop_hash << "\n"
                << "the hash in mindir: " << hash_in_mindir;

  return false;
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
  MindIRLoader mindir_loader;
  auto bprop_fg = mindir_loader.LoadMindIR(bprop_mindir_realpath.value());
  if (bprop_fg == nullptr) {
    MS_LOG(WARNING) << "Failed to load the bprop mindir " << bprop_mindir_realpath.value();
    return nullptr;
  }
  if (!CheckBpropHash(prim, bprop_fg->bprop_hash(), bprop_fg->bprop_filepath())) {
    MS_LOG(EXCEPTION) << "The bprop mindir files are not up to date.";
  }
  return bprop_fg;
}

void EliminateParameterSelf(const FuncGraphPtr &func_graph, const pipeline::ResourceBasePtr &resources,
                            const PrimitivePtr &prim) {
  auto parameters = func_graph->parameters();
  if (parameters.empty()) {
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
  static const auto all_bprop_to_mindir = (common::GetEnv("MS_DEV_ALL_BPROP_TO_MINDIR") == "1");
  auto res = (resources != nullptr) ? resources : std::make_shared<pipeline::Resource>();
  if (all_bprop_to_mindir) {
    EliminateParameterSelf(fg, res, prim);
  }
  opt::irpass::BpropMindIRPassLib irpass;
  std::vector<opt::SubstitutionPtr> opt_list{irpass.get_multitype_ops_};
  if (all_bprop_to_mindir) {
    (void)opt_list.emplace_back(irpass.get_class_type_);
    (void)opt_list.emplace_back(irpass.get_meta_fg_);
    (void)opt_list.emplace_back(irpass.get_primal_attr_);
  }
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
    {"remove_class_type", opt::OptPassConfig({irpass.remote_class_type_})},
  });
  opt::OptimizerPtr export_mindir_opt =
    opt::Optimizer::MakeOptimizer("export_mindir_opt", res, map, false, false, false);
  (void)export_mindir_opt->step(func_graph, false);
  return func_graph;
}
#endif
}  // namespace

#ifndef _WIN32
void ExportBpropToMindir(const py::object &obj) {
  std::string prim_name;
  if (!py::isinstance<py::str>(obj)) {
    py::object obj_name = obj.attr("__name__");
    if (!py::isinstance<py::str>(obj_name)) {
      MS_LOG(EXCEPTION) << "The name of obj " << py::str(obj) << " to be exported to mindir should be a string";
    }
    prim_name = obj_name.cast<std::string>();
  } else {
    prim_name = obj.cast<std::string>();
  }
  // Get the bprop function from python.
  py::function fn = GetBpropFunctionByObj(obj);
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
  if (!NeedExportBpropMindIR(prim_name, bprop_hash_and_filepath.first)) {
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

  static const auto all_bprop_to_mindir = (common::GetEnv("MS_DEV_ALL_BPROP_TO_MINDIR") == "1");
  if (all_bprop_to_mindir) {
    func_graph = LiftParameter(func_graph);
    func_graph = RemovePyObj(func_graph, res);
  }
  func_graph->set_bprop_hash(bprop_hash_and_filepath.first);
  func_graph->set_bprop_filepath(bprop_hash_and_filepath.second);
  if (!ExportBpropToMindIR(prim_name, func_graph)) {
    MS_LOG(EXCEPTION) << "Failed to export the bprop mindir for " << prim_name;
  }
}
#endif

FuncGraphPtr GetBprop(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &resources) {
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
  static const auto all_bprop_to_mindir = (common::GetEnv("MS_DEV_ALL_BPROP_TO_MINDIR") == "1");
  bool serializable = (all_bprop_to_mindir || IsSerializableBprop(prim->name()));
  if (serializable) {
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
#ifndef _WIN32
  // Check whether the bprop needs to be exported.
  if (serializable) {
    auto bprop_hash_and_filepath = GetBpropHashAndFilePath(fn, prim->name());
    if (!bprop_hash_and_filepath.first.empty()) {
      func_graph->set_bprop_hash(bprop_hash_and_filepath.first);
      func_graph->set_bprop_filepath(bprop_hash_and_filepath.second);
      if (!ExportBpropToMindIR(prim->name(), func_graph)) {
        MS_LOG(WARNING) << "Failed to export the bprop mindir for " << prim->name();
      }
    }
  }
#endif
  return func_graph;
}
}  // namespace ad
}  // namespace mindspore
