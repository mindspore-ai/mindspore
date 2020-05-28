/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef TESTS_UT_COMMON_PY_FUNC_GRAPH_FETCHER_H_
#define TESTS_UT_COMMON_PY_FUNC_GRAPH_FETCHER_H_

#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/manager.h"
#include "ir/func_graph.h"
#include "pipeline/parse/parse_base.h"
#include "pipeline/parse/parse.h"
#include "./common.h"

namespace UT {

void InitPythonPath();

class PyFuncGraphFetcher {
 public:
  explicit PyFuncGraphFetcher(std::string model_path, bool doResolve = false)
      : model_path_(model_path), doResolve_(doResolve) {
    InitPythonPath();
  }
  void SetDoResolve(bool doResolve = true) { doResolve_ = doResolve; }

  // The return of python function of "func_name" should be py::function.
  // step 1. Call the function user input
  // step 2. Parse the return "fn"
  template <class... T>
  mindspore::FuncGraphPtr CallAndParseRet(std::string func_name, T... args) {
    try {
      py::function fn = mindspore::parse::python_adapter::CallPyFn(model_path_.c_str(), func_name.c_str(), args...);
      mindspore::FuncGraphPtr func_graph = mindspore::parse::ParsePythonCode(fn);
      if (doResolve_) {
        std::shared_ptr<mindspore::FuncGraphManager> manager = mindspore::Manage(func_graph, false);
        mindspore::parse::python_adapter::set_use_signature_in_resolve(false);
        mindspore::parse::ResolveAll(manager);
      }
      return func_graph;
    } catch (py::error_already_set& e) {
      MS_LOG(ERROR) << "Call and parse fn failed!!! error:" << e.what();
      return nullptr;
    } catch (...) {
      MS_LOG(ERROR) << "Call fn failed!!!";
      return nullptr;
    }
  }

  // Fetch python function then parse to graph
  mindspore::FuncGraphPtr operator()(std::string func_name, std::string model_path = "") {
    try {
      std::string path = model_path_;
      if ("" != model_path) {
        path = model_path;
      }
      py::function fn = mindspore::parse::python_adapter::GetPyFn(path.c_str(), func_name.c_str());
      mindspore::FuncGraphPtr func_graph = mindspore::parse::ParsePythonCode(fn);
      if (doResolve_) {
        std::shared_ptr<mindspore::FuncGraphManager> manager = mindspore::Manage(func_graph, false);
        mindspore::parse::ResolveAll(manager);
      }
      return func_graph;
    } catch (py::error_already_set& e) {
      MS_LOG(ERROR) << "get fn failed!!! error:" << e.what();
      return nullptr;
    } catch (...) {
      MS_LOG(ERROR) << "get fn failed!!!";
      return nullptr;
    }
  }

 private:
  std::string model_path_;
  bool doResolve_;
};

}  // namespace UT
#endif  // TESTS_UT_COMMON_PY_FUNC_GRAPH_FETCHER_H_
