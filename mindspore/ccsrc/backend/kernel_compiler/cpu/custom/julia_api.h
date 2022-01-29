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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CUSTOM_JULIA_API_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CUSTOM_JULIA_API_H_

#if !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#endif

#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <condition_variable>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr auto kLibJulia = "libjulia.so";
typedef struct _jl_value_t jl_value_t;
typedef jl_value_t jl_function_t;
typedef struct _jl_module_t jl_module_t;
typedef struct _jl_datatype_t jl_datatype_t;
typedef struct _jl_array_t jl_array_t;
typedef struct _jl_sym_t jl_sym_t;

#define GET_HOOK(func, rt, ...) GET_HOOK_INNER(func, _, rt, __VA_ARGS__)
#define GET_HOOK_INNER(func, _, rt, ...)                                                          \
  do {                                                                                            \
    if (!func##_) {                                                                               \
      func##_ = reinterpret_cast<std::add_pointer<rt(__VA_ARGS__)>::type>(dlsym(handle_, #func)); \
      if (auto error_info = dlerror(); error_info != nullptr) {                                   \
        MS_LOG(EXCEPTION) << error_info;                                                          \
        suc = false;                                                                              \
      }                                                                                           \
    }                                                                                             \
  } while (0)

/*
 *   kernel one             julia thread              kernel two                JuliaAPI
 *   =============init================
 *   lock                                              try lock
 *   init       ---------> create, loop, try lock      try lock                dlopen/dlsym
 *   wait       --------->     lock                    try lock
 *   wait                      init                    try lock
 *   wait       <--------- notify inited               try lock
 *   unlock     <---------     wait                    try lock
 *   ==========init finish============
 *                              ===============init================
 *   try lock                  wait                      lock
 *   try lock                  wait                      init
 *   try lock                  wait                     unlock
 *                              =============init finish===========
 *   ==============run================
 *   lock                      wait                    try lock
 *   running=true, notify -->  unlock                  try lock
 *   wait                      try lock                  lock
 *   wait                      try lock             another running, wait
 *   wait                      lock                 another running, wait
 *   wait                      run                  another running, wait
 *   wait       <--------- notify run finished ---> another running, wait
 *   running=false, result     wait                    try lock
 *   unlock                    wait                    try lock
 *   ===========run finish============
 *                               ===============run================
 *                                                     lock
 *                             unlock       <-------   running=true, notify run
 *                             lock                    wait
 *                             run                     wait
 *                         notify run finished ----->  wait
 *                             wait                   get result
 *                               ============run finish============
 *                             stop                                            destructor
 *                             join                                             dl close
 */
class JuliaAPI {
 public:
  static JuliaAPI *GetInstance() {
    static JuliaAPI julia_api;
    return &julia_api;
  }

  bool Init() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (handle_) {
      // only init once
      return true;
    } else {
#if !defined(_WIN32) && !defined(_WIN64)
      // open julia shared library
      handle_ = dlopen(kLibJulia, RTLD_LAZY | RTLD_LOCAL);
      if (!handle_) {
        MS_LOG(EXCEPTION) << dlerror();
        return false;
      }
#else
      return false;
#endif
    }
    // get julia func from library
    if (!InitJuliaFunc()) {
      return false;
    }
    // create a thread to call julia func, the thread will exist until JuliaAPI instance gone
    t_ = std::thread(Loop);
    // wait for the loop in the thread begin and julia init finished
    c_.wait(lock);
    // when init finished, keep initialized_ true
    initialized_ = true;
    return true;
  }

  int Run(const std::string &file, const std::string &module, const std::string &func, int nparam,
          const std::vector<void *> &param, const std::vector<int> &ndims, const std::vector<int64_t *> &shapes,
          const std::vector<const char *> &dtypes) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto NotRunning = [this]() { return !this->running_; };
    // when second kernel get mutex should wait first kernel run finished
    c_.wait(lock, NotRunning);
    // when running a kernel, set it to true
    running_ = true;
    // set julia kernel's inputs
    file_ = file;
    module_ = module;
    func_ = func;
    nparam_ = nparam;
    params_ = param;
    ndims_ = ndims;
    shapes_ = shapes;
    dtypes_ = dtypes;

    // notify the loop continue, which will run the kernel
    c_.notify_one();
    // wait for the kernel run finish
    c_.wait(lock);
    // when the kernel run finished, set it to false again
    running_ = false;
    return result_;
  }

 private:
  JuliaAPI() {
    handle_ = nullptr;
    jl_eval_string_ = nullptr;
    jl_get_global_ = nullptr;
    jl_symbol_ = nullptr;
    jl_call_ = nullptr;
    jl_exception_occurred_ = nullptr;
    jl_atexit_hook_ = nullptr;
    jl_init__threading_ = nullptr;
    jl_apply_array_type_ = nullptr;
    jl_ptr_to_array_ = nullptr;
    jl_typeof_str_ = nullptr;
    jl_stderr_obj_ = nullptr;
    jl_current_exception_ = nullptr;
    jl_ver_major_ = nullptr;
    jl_ver_minor_ = nullptr;
  }
  ~JuliaAPI() {
#if !defined(_WIN32) && !defined(_WIN64)
    if (handle_ != nullptr) {
      // ready to break the loop in the julia thread
      stop_ = true;
      // notify loop continue, which will stop the loop, then julia thread finished.
      c_.notify_one();
      // join the julia thread
      t_.join();
      // close the handle of julia shared library
      dlclose(handle_);
    }
#endif
  }

  static void Loop() {
    // keep the thread alive, for the julia runtime should run in one thread safely
    while (true) {
      // 1. init julia runtime by first time
      // 2. wait for julia kernel run
      // 3. run the julia kernel when get the signal of Run
      GetInstance()->RunInJuliaThread();
      // stopped by JuliaAPI instance's destructor
      if (GetInstance()->stop_) {
        break;
      }
    }
  }

  void RunInJuliaThread() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (initialized_) {
      // run julia kernel when get signal from Run
      result_ = RunJuliaKernel();
    } else {
      // init julia runtime by first time
      JlInit();
    }

    // notify to Init by first time
    // notify to Run that kernel run finished by next time
    // or notify to another kernel run that this kernel run finished
    c_.notify_all();
    // wait for the signal of Run
    c_.wait(lock);
  }

  bool InitJuliaFunc() {
    bool suc = true;
#if !defined(_WIN32) && !defined(_WIN64)
    GET_HOOK(jl_ver_major, int, void);
    GET_HOOK(jl_ver_minor, int, void);
    if (!suc) return false;
    constexpr int SupportedMinor = 6;
    if (JlVerMajor() < 1 || (JlVerMajor() == 1 && JlVerMinor() < SupportedMinor)) {
      MS_LOG(WARNING) << "we only support julia version >= 1.6 now and have tested in version 1.6";
      return false;
    }
    GET_HOOK(jl_eval_string, jl_value_t *, const char *);
    GET_HOOK(jl_get_global, jl_value_t *, jl_module_t *, jl_sym_t *);
    GET_HOOK(jl_symbol, jl_sym_t *, const char *);
    GET_HOOK(jl_call, jl_value_t *, jl_function_t *, jl_value_t **, int32_t);
    GET_HOOK(jl_exception_occurred, jl_value_t *, void);
    GET_HOOK(jl_atexit_hook, void, int);
    GET_HOOK(jl_init__threading, void, void);
    GET_HOOK(jl_apply_array_type, jl_value_t *, jl_value_t *, size_t);
    GET_HOOK(jl_ptr_to_array, jl_array_t *, jl_value_t *, void *, jl_value_t *, int);
    GET_HOOK(jl_typeof_str, const char *, jl_value_t *);
    GET_HOOK(jl_stderr_obj, jl_value_t *, void);
    GET_HOOK(jl_current_exception, jl_value_t *, void);
#else
    suc = false;
#endif
    return suc;
  }

  int RunJuliaKernel() {
    // include julia file
    JlEvalString("Base.include(Main, \"" + file_ + "\")");
    // using julia module
    JlEvalString("using Main." + module_);
    jl_module_t *jmod = reinterpret_cast<jl_module_t *>(JlEvalString("Main." + module_));
    // get julia function from module
    jl_function_t *jfunc = JlGetFunction(jmod, func_);
    // convert kernel inputs to julia type
    std::vector<jl_value_t *> args(nparam_);
    for (int i = 0; i < nparam_; i++) {
      args[i] = reinterpret_cast<jl_value_t *>(GetJuliaArray(params_[i], ndims_[i], shapes_[i], dtypes_[i]));
    }
    // call the julia function
    JlCall(jfunc, &args[0], nparam_);
    if (JlExceptionOccurred()) {
      MS_LOG(EXCEPTION) << JlTypeOfStr(JlExceptionOccurred());
      auto errs = JlStdErrObj();
      if (errs) {
        JlEvalString("using Main.Base");
        auto base = reinterpret_cast<jl_module_t *>(JlEvalString("Main.Base"));
        auto show = JlGetFunction(base, "show");
        if (show) {
          std::vector<jl_value_t *> err_args{errs, JlCurrentException()};
          constexpr int arg_num = 2;
          JlCall(show, &err_args[0], arg_num);
        }
      }
      return -1;
    }
    JlAtexitHook(0);
    return 0;
  }

  jl_value_t *JlEvalString(const std::string &str) { return jl_eval_string_(str.c_str()); }

  jl_value_t *JlGetGlobal(jl_module_t *m, jl_sym_t *var) { return jl_get_global_(m, var); }

  jl_sym_t *JlSymbol(const std::string &str) { return jl_symbol_(str.c_str()); }

  jl_value_t *JlCall(jl_function_t *f, jl_value_t **args, int32_t nargs) { return jl_call_(f, args, nargs); }

  jl_value_t *JlExceptionOccurred(void) { return jl_exception_occurred_(); }

  void JlAtexitHook(int status) { return jl_atexit_hook_(status); }

  void JlInit(void) { return jl_init__threading_(); }

  jl_value_t *JlApplyArrayType(jl_value_t *type, size_t dim) { return jl_apply_array_type_(type, dim); }

  jl_array_t *JlPtrToArray(jl_value_t *atype, void *data, jl_value_t *dims, int own_buffer) {
    return jl_ptr_to_array_(atype, data, dims, own_buffer);
  }

  const char *JlTypeOfStr(jl_value_t *v) { return jl_typeof_str_(v); }

  jl_value_t *JlStdErrObj() { return jl_stderr_obj_(); }

  jl_value_t *JlCurrentException() { return jl_current_exception_(); }

  int JlVerMajor() { return jl_ver_major_(); }

  int JlVerMinor() { return jl_ver_minor_(); }

  jl_function_t *JlGetFunction(jl_module_t *m, const std::string &name) {
    return reinterpret_cast<jl_function_t *>(JlGetGlobal(m, JlSymbol(name)));
  }

  jl_value_t *Core(const std::string &name) {
    jl_module_t *jl_core_module = reinterpret_cast<jl_module_t *>(JlEvalString("Main.Core"));
    return JlGetGlobal(jl_core_module, JlSymbol(name.c_str()));
  }

  jl_datatype_t *GetType(const std::string &dtypes) {
    jl_datatype_t *type = reinterpret_cast<jl_datatype_t *>(Core("Float32"));
    std::unordered_map<std::string, std::string> m{
      {"float16", "Float16"}, {"float32", "Float32"}, {"float64", "Float64"}, {"int8", "Int8"},
      {"uint8", "UInt8"},     {"int16", "Int16"},     {"uint16", "Uint16"},   {"int32", "Int32"},
      {"uint32", "UInt32"},   {"int64", "Int64"},     {"uint64", "UInt64"}};
    if (m.count(dtypes)) {
      type = reinterpret_cast<jl_datatype_t *>(Core(m[dtypes]));
    }
    return type;
  }

  jl_array_t *GetJuliaArray(void *params, int ndims, int64_t *shapes, const std::string &dtypes) {
    std::string shape_str = "(";
    for (int j = 0; j < ndims; j++) {
      shape_str += std::to_string(shapes[j]);
      shape_str += ",";
    }
    shape_str += ")";
    auto shape = JlEvalString(shape_str);
    auto dtype = reinterpret_cast<jl_value_t *>(GetType(dtypes));
    auto array_type = JlApplyArrayType(dtype, ndims);
    return JlPtrToArray(array_type, params, shape, 0);
  }

 private:
  // the thread which used to call julia func, will be created by first julia kernel,
  // and will always exist until the JuliaAPI instance gone.
  std::thread t_;
  // the mutex and condition_variable is used to control the workflow and keep thread safe.
  std::mutex mutex_;
  std::condition_variable c_;
  // when stop is true, the loop in the thread will break
  bool stop_{false};
  // when dlopen success, dlsym success, created thread and loop in the thread started, init can be true.
  bool initialized_{false};
  // check if a kernel is running, another kernel should wait if a kernel is already running
  bool running_{false};
  // the result show julia kernel run success or not.
  int result_{0};

  // julia kernel's inputs
  std::vector<void *> params_;
  int nparam_{0};
  std::string file_;
  std::string module_;
  std::string func_;
  std::vector<int> ndims_;
  std::vector<int64_t *> shapes_;
  std::vector<const char *> dtypes_;

  // about julia shared library
  void *handle_;
  jl_value_t *(*jl_eval_string_)(const char *);
  jl_value_t *(*jl_get_global_)(jl_module_t *, jl_sym_t *);
  jl_sym_t *(*jl_symbol_)(const char *);
  jl_value_t *(*jl_call_)(jl_function_t *, jl_value_t **, int32_t);
  jl_value_t *(*jl_exception_occurred_)(void);
  void (*jl_atexit_hook_)(int);
  void (*jl_init__threading_)(void);
  jl_value_t *(*jl_apply_array_type_)(jl_value_t *, size_t);
  jl_array_t *(*jl_ptr_to_array_)(jl_value_t *, void *, jl_value_t *, int);
  const char *(*jl_typeof_str_)(jl_value_t *);
  jl_value_t *(*jl_stderr_obj_)(void);
  jl_value_t *(*jl_current_exception_)(void);
  int (*jl_ver_major_)(void);
  int (*jl_ver_minor_)(void);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CUSTOM_JULIA_API_H_
