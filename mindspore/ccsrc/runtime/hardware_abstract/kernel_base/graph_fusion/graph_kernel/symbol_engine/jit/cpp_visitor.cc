/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/symbol_engine/jit/cpp_visitor.h"

#if !(defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#include <dlfcn.h>

#include <array>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>

#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "include/utils/common.h"
#include "utils/file_utils.h"

namespace mindspore::graphkernel::symshape {
constexpr const char *kPrefix = "symbol_engine_jit_";
using ast::Shape, ast::BinOpType;

static string KernelMetaPath() {
  static string result = "";
  if (result != "") {
    return result;
  }
  auto config_path = kernel::GetCompilerCachePath();
  auto kernel_meta_path = config_path + std::string(kernel::kAkgKernelMeta);
  auto real_path = FileUtils::GetRealPath(kernel_meta_path.c_str());
  if (!real_path.has_value()) {
    MS_LOG(EXCEPTION) << "get real path failed: " << kernel_meta_path;
  }
  result = real_path.value() + "/";
  return result;
}

CppVisitor::CppVisitor() {
  time_t t = time(nullptr);
  name_ = "cppvisitor_" + std::to_string(t);
}

CppVisitor::~CppVisitor() {
  if (dynlib_) {
    dlclose(dynlib_);
  }
}

std::string CppVisitor::CodeGen(const std::vector<ast::ShapePtr> &shapes, const ast::SymbolTable &symbol_table,
                                const std::string &func_name) {
  symbols_table_ = &symbol_table;
  static int64_t func_idx = 1;
  std::stringstream func;
  std::string final_func_name = "func_" + std::to_string(func_idx) + "_" + func_name;
  func_idx++;
  var_tag_ = std::vector<int32_t>(symbols_table_->size(), 0);
  // function implementation
  func << "extern \"C\" void " << final_func_name << "(const int64_t **input, int64_t** res){\n";
  //  assemble shape expression
  std::stringstream res_expr;
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto &shape = shapes[i];
    for (size_t j = 0; j < shape->smbls_.size(); ++j) {
      shape->smbls_[j]->Accept(this);
      res_expr << "res[" << i << "][" << j << "] = " << cpp_sentences_.back() << ";\n";
      cpp_sentences_.pop_back();
    }
  }
  cpp_sentences_.push_back(res_expr.str());
  for (auto &sentence : cpp_sentences_) {
    func << sentence << '\n';
  }
  func << "\n}\n";
  func_blocks_.push_back(func.str());
  // clear shape_ and cpp_sentence;
  cpp_sentences_.clear();
  symbols_table_ = nullptr;
  var_tag_.clear();
  null_ = false;

  return final_func_name;
}

void CppVisitor::Compile() {
  if (null_) {
    // skip compile if no function is generated
    return;
  }
  MS_LOG(DEBUG) << "Start to compile cpp file used to infer shape";
  compile_thread_ = std::thread(&CppVisitor::CompileImpl, this);
}

void CppVisitor::CompileImpl() {
  auto kernel_meta_path = KernelMetaPath();
  (void)FileUtils::CreateNotExistDirs(kernel_meta_path);
  std::string cpp_file_name(kernel_meta_path + kPrefix + name_ + ".cc");
  auto real_filename = FileUtils::GetRealPath(cpp_file_name.c_str());
  if (!real_filename.has_value()) {
    MS_LOG(EXCEPTION) << "Failed to get real name for " << cpp_file_name;
  }
  MS_LOG(DEBUG) << "SymbolEngineJit c++ function saved to: " << real_filename.value();
  std::ofstream cpp_file(real_filename.value());

  // --- generate  .cc file
  const string header = R"(
#include <vector>
#include <cstdint>

)";

  cpp_file << header;
  for (auto &func : func_blocks_) {
    cpp_file << func << "\n";
  }
  cpp_file.close();

  // compile to dyn lib
  std::stringstream cmd;
  std::string so_name(kernel_meta_path + kPrefix + name_ + ".so");
  cmd << "g++ -fPIC -shared -std=c++17 " << real_filename.value() << " -o " << so_name << " 2>&1";

  // create library
  constexpr size_t kBufferSize = 256;
  std::array<char, kBufferSize> buffer{};
  string result;
  FILE *pipe = popen(cmd.str().c_str(), "r");
  if (!pipe) {
    MS_LOG(EXCEPTION) << "fail to run command to compile c++ code, error:" << strerror(errno);
    return;
  }
  while (fgets(buffer.data(), kBufferSize, pipe)) {
    result += buffer.data();
  }
  void(pclose(pipe));
  if (!Common::FileExists(so_name)) {
    MS_LOG(EXCEPTION) << "compile failed: no .so file: " << so_name << "\n Information from pipe: " << result;
  }
  MS_LOG(DEBUG) << "Finished compiling, information from pipe: \n" << result;
}

CppVisitor::DynFuncType CppVisitor::LoadFunc(const std::string &func_name) {
  MS_LOG(DEBUG) << "CppVisitor trying to load function: " << func_name;
  if (compile_thread_.joinable()) {
    compile_thread_.join();
  }
  if (!dynlib_) {
    auto so_name = FileUtils::GetRealPath((KernelMetaPath() + kPrefix + name_ + ".so").c_str());
    if (!so_name.has_value()) {
      MS_LOG(EXCEPTION) << "Failed to get real path for " << KernelMetaPath() << kPrefix << name_ << ".so";
    }
    dynlib_ = dlopen(so_name.value().c_str(), RTLD_LAZY);
    if (!dynlib_) {
      MS_LOG(EXCEPTION) << "Cannot open dynamic library " << so_name.value() << ".so :" << dlerror() << '\n';
    }
  }

  auto fn = (DynFuncType)(dlsym(dynlib_, func_name.c_str()));
  if (!fn) {
    MS_LOG(EXCEPTION) << "Cannot find function " << func_name << " :" << dlerror() << '\n';
  }
  return fn;
}

void CppVisitor::Visit(const ast::IntImm &imm) { cpp_sentences_.push_back(std::to_string(imm.shape_int)); }

void CppVisitor::Visit(const ast::BinOp &op) {
  std::stringstream sentence;

  std::string ope_string = "";
  bool prefix = true;

  switch (op.optype_) {
    case BinOpType::ScalarMax:
      ope_string = "std::max";
      break;
    case BinOpType::ScalarMin:
      ope_string = "std::min";
      break;
    case BinOpType::ScalarDiv:
      ope_string = "/";
      prefix = false;
      break;
    case BinOpType::ScalarAdd:
      ope_string = "+";
      prefix = false;
      break;
    case BinOpType::ScalarSub:
      ope_string = "-";
      prefix = false;
      break;
    case BinOpType::ScalarMul:
      ope_string = "*";
      prefix = false;
      break;
    default:
      MS_LOG(EXCEPTION) << "Unexpected operation";
      break;
  }

  if (prefix) {
    sentence << ope_string << "(";
    op.a_->Accept(this);
    sentence << cpp_sentences_.back() << ", ";
    cpp_sentences_.pop_back();
    op.b_->Accept(this);
    sentence << cpp_sentences_.back() << ")";
    cpp_sentences_.pop_back();

    cpp_sentences_.push_back(sentence.str());
  } else {
    op.a_->Accept(this);
    sentence << cpp_sentences_.back() << ope_string;
    cpp_sentences_.pop_back();
    op.b_->Accept(this);
    sentence << cpp_sentences_.back();
    cpp_sentences_.pop_back();
    cpp_sentences_.push_back(sentence.str());
    return;
  }
}

void CppVisitor::Visit(const ast::Var &input_smbl) {
  if (var_tag_[input_smbl.id_]) {
    cpp_sentences_.push_back(input_smbl.ToString());
    return;
  }

  var_tag_[input_smbl.id_] = 1;
  // assume no recurive call
  auto smbl_p = (*symbols_table_)[input_smbl.id_];
  std::stringstream sentence;
  sentence << "int64_t " << input_smbl.ToString() << " = ";
  smbl_p->Accept(this);
  sentence << cpp_sentences_.back() << ";";
  cpp_sentences_.pop_back();
  cpp_sentences_.push_back(sentence.str());
  cpp_sentences_.push_back(input_smbl.ToString());
}

void CppVisitor::Visit(const ast::Input &input_smbl) {
  std::stringstream sentence;
  sentence << "input[" << input_smbl.i_ << "][" << input_smbl.j_ << "]";
  cpp_sentences_.push_back(sentence.str());
}

void CppVisitor::Visit(const ast::Shape &shape) {
  MS_LOG(DEBUG) << "CppVisitor Visit a Shape: " << shape.ToString();
  std::stringstream sentence;
  std::string name = UniqueName();
  sentence << "std::vector<int64_t> " << name << " {";

  for (auto smbl : shape.smbls_) {
    smbl->Accept(this);
    sentence << cpp_sentences_.back() << ", ";
    cpp_sentences_.pop_back();
  }
  if (!shape.smbls_.empty()) {
    // remove the last ", "
    constexpr int remove_len = 2;
    sentence.seekp(-remove_len, sentence.cur);
  }

  sentence << "};";

  cpp_sentences_.push_back(sentence.str());
  cpp_sentences_.push_back(name);
}
}  // namespace mindspore::graphkernel::symshape
#else
namespace mindspore::graphkernel::symshape {
using ast::Shape, ast::BinOpType;
CppVisitor::CppVisitor() {}
CppVisitor::~CppVisitor() {}
std::string CppVisitor::CodeGen(const std::vector<ast::ShapePtr> &, const ast::SymbolTable &, const std::string &) {
  return "";
}
void CppVisitor::Compile() {}
CppVisitor::DynFuncType CppVisitor::LoadFunc(const std::string &) { return nullptr; }
void CppVisitor::Visit(const ast::IntImm &imm) {}
void CppVisitor::Visit(const ast::BinOp &op) {}
void CppVisitor::Visit(const ast::Var &input_smbl) {}
void CppVisitor::Visit(const ast::Input &input_smbl) {}
void CppVisitor::Visit(const ast::Shape &shape) {}
}  // namespace mindspore::graphkernel::symshape
#endif
