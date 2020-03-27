/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this ${file} except in compliance with the License.
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

#ifndef LITE_RUNTIME_ON

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <shared_mutex>
#include <memory>
#include "common/mslog.h"

const char *LIB_INFO = "libtvm_kernel version: master (c66c6b28dc991c9d705e1b983aab7385c337128d)";
namespace km {
class KernelManager {
 public:
  int CallKernel(const std::string &fid, TVMArgs args) {
    tvm::runtime::Module *mod = this->GetModule();
    CHECK(mod != nullptr) << "Failed to get Module!";
    const std::string name = fid;
    tvm::runtime::PackedFunc f = mod->GetFunction(name, false);
    CHECK(f != nullptr) << "Can't find kernel func " << fid;
    TVMRetValue rv;
    f.CallPacked(args, &rv);
    return 0;
  }

  void InitKernelManager(int mode, const std::string &fname) { return this->Init(mode, fname); }

  static KernelManager *Global() {
    static KernelManager inst;
    return &inst;
  }

  tvm::runtime::Module *GetModule() const { return &g_modLib; }

 private:
  KernelManager() = default;

  ~KernelManager() = default;

  void Init(int mode, std::string fpath) {
    std::call_once(init_flag, &KernelManager::InitLib, mode, fpath);
    return;
  }

  static void InitLib(int mode, std::string fpath) {
    if (mode) {
      const PackedFunc *ptr = tvm::runtime::Registry::Get("module._GetSystemLib");
      CHECK(ptr != nullptr) << "Failed to get systemlib";
      g_modLib = (*ptr)();
    } else {
      g_modLib = tvm::runtime::Module::LoadFromFile(fpath);
    }
  }
  static tvm::runtime::Module g_modLib;
  std::once_flag init_flag;
};

tvm::runtime::Module KernelManager::g_modLib;
}  // namespace km

std::function<int(const std::vector<DLTensor *> &)> GetKernel(const std::string &fid) {
  km::KernelManager *inst = km::KernelManager::Global();
  CHECK(inst != nullptr) << "Failed to get KernelManager instance!";
  tvm::runtime::Module *mod = inst->GetModule();
  CHECK(mod != nullptr) << "Failed to get Module!";
  tvm::runtime::PackedFunc f = mod->GetFunction(fid, false);
  if (f == nullptr) {
    MS_LOGE("GetFunction return nullptr");
    return nullptr;
  }
  auto runner = [f](const std::vector<DLTensor *> &tensors) -> int {
    int argLen = tensors.size();
    CHECK(argLen) << "Input tensors num=0 !";
    std::vector<TVMValue> values(argLen);
    std::vector<int> codes(argLen);
    tvm::runtime::TVMArgsSetter setter(values.data(), codes.data());
    for (int i = 0; i < argLen; ++i) {
      setter(i, tensors.at(i));
    }
    tvm::runtime::TVMArgs targs(values.data(), codes.data(), argLen);
    TVMRetValue rv;
    f.CallPacked(targs, &rv);
    return 0;
  };
  return runner;
}

int CallKernel(const std::string &fid, const std::vector<DLTensor *> &tensors) {
  km::KernelManager *inst = km::KernelManager::Global();
  CHECK(inst != nullptr) << "Failed to get KernelManager instance!";
  int argLen = tensors.size();
  CHECK(argLen) << "Input tensors num=0 !";
  std::vector<TVMValue> values(argLen);
  std::vector<int> codes(argLen);
  tvm::runtime::TVMArgsSetter setter(values.data(), codes.data());
  for (int i = 0; i < argLen; ++i) {
    setter(i, tensors.at(i));
  }
  tvm::runtime::TVMArgs targs(values.data(), codes.data(), argLen);
  inst->CallKernel(fid, targs);
  return 0;
}

int InitKernelManager(int mode, const std::string &fname) {
  km::KernelManager *inst = km::KernelManager::Global();
  CHECK(inst != nullptr) << "Failed to get KernelManager instance!";
  inst->InitKernelManager(mode, fname);
  return 0;
}

// just for api compatible, tvm/lite has same api
void ConfigThreadPool(int mode = 1, int nthreads = 0, bool execute_self = true) {}

#else

#include <lite/api/km_api.h>
#include <tvm/runtime/packed_func.h>
#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include "flatbuffers/flatbuffers.h"
#include "schema/inner/ms_generated.h"
#include "include/securec.h"
#include "src/runtime/runtime_api.h"
#include "common/mslog.h"

using runnerType = std::function<int(const std::vector<DLTensor *> &)>;

const char *LIB_INFO = "libtvm_kernel version: master (c66c6b28dc991c9d705e1b983aab7385c337128d)";

namespace lite {
namespace runtime {
extern "C" {
// Function signature for generated packed function in shared library
typedef int (*BackendPackedCFunc)(const void *args, int *type_codes, int num_args);
}  // extern "C"

class LiteFuncPool {
 public:
  LiteFuncPool() = default;

  ~LiteFuncPool() = default;

  void GetFunction(const std::string &name, void **func_addr) {
    auto it = tbl_.find(name);
    if (func_addr == nullptr) {
      MS_LOGW("input func_addr is nullptr");
      return;
    }
    *func_addr = (it != tbl_.end() ? it->second : nullptr);
  }

  void RegisterSymbol(const std::string &name, void *ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tbl_.find(name);
    if (it != tbl_.end() && ptr != it->second) {
      MS_LOGW("Lite symbol %s get overriden to a different address %p->%p", name.c_str(), ptr, it->second);
    }
    tbl_[name] = ptr;
  }

  static LiteFuncPool *Global() {
    static LiteFuncPool inst;
    return &inst;
  }

 private:
  // Internal mutex
  std::mutex mutex_;
  // Internal symbol table
  std::unordered_map<std::string, void *> tbl_;
};
}  // namespace runtime
}  // namespace lite

using LiteFuncPool = lite::runtime::LiteFuncPool;
using BackendPackedCFunc = lite::runtime::BackendPackedCFunc;

int LiteBackendRegisterSystemLibSymbol(const char *name, void *ptr) {
  MS_ASSERT(LiteFuncPool::Global() != nullptr);
  LiteFuncPool::Global()->RegisterSymbol(name, ptr);
  return 0;
}

// do nothing, api compatible with TVM_RUNTIME_ON API
void InitKernelManager(int mode, const std::string &fname) { return; }

static inline void *GetFunction(const std::string &fid) {
  void *f = nullptr;
  MS_ASSERT(LiteFuncPool::Global() != nullptr);
  LiteFuncPool::Global()->GetFunction(fid, &f);
  if (f == nullptr) {
    return nullptr;
  }
  return f;
}

runnerType __attribute__((noinline)) GetKernel(const std::string &fid) {
  auto f = GetFunction(fid);
  if (f == nullptr) {
    return nullptr;
  }
  auto runner = [f](const std::vector<DLTensor *> &tensors) -> int {
    if (tensors.empty()) {
      MS_LOGE("Input tensors num = 0 !");
      return -1;
    }
    std::vector<TVMValue> values(tensors.size());
    std::vector<int> codes(tensors.size());
    tvm::runtime::TVMArgsSetter setter(values.data(), codes.data());
    for (size_t i = 0; i < tensors.size(); ++i) {
      setter(i, tensors.at(i));
    }
    auto passfunc = reinterpret_cast<BackendPackedCFunc>(f);
    return passfunc(values.data(), codes.data(), tensors.size());
  };
  return runner;
}

namespace auto_tensor {
constexpr int TENSOR_NUM_MAX = 10;
constexpr bool STORE_MODE = true;
constexpr bool RESUME_MODE = false;
const char *NOT_SUPPORT = "NOT SUPPORT";
const int NCHW_N = 0;
const int NCHW_C = 1;
const int NCHW_H = 2;
const int NCHW_W = 3;
const int tile = 4;

void store_shape(const std::vector<DLTensor *> &tensors, int (&ndim)[TENSOR_NUM_MAX], int64_t *(&shape)[TENSOR_NUM_MAX],
                 int64_t *(&strides)[TENSOR_NUM_MAX], bool mode = STORE_MODE) {
  if (mode == STORE_MODE) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      ndim[i] = tensors[i]->ndim;
      shape[i] = tensors[i]->shape;
      strides[i] = tensors[i]->strides;
    }
  } else {
    for (size_t i = 0; i < tensors.size(); ++i) {
      tensors[i]->ndim = ndim[i];
      tensors[i]->shape = shape[i];
      tensors[i]->strides = strides[i];
    }
  }
}

static std::string get_dtype(const DLTensor &tensor) {
  auto dtype = tensor.dtype;
  if (dtype.code == kDLFloat) {
    if (dtype.bits == 16)
      return "float16";
    else if (dtype.bits == 32)
      return "float32";
    else if (dtype.bits == 64)
      return "float64";
  } else if (dtype.code == kDLInt) {
    if (dtype.bits == 8)
      return "int8";
    else if (dtype.bits == 16)
      return "int16";
    else if (dtype.bits == 32)
      return "int32";
    else if (dtype.bits == 64)
      return "int64";
  } else if (dtype.code == kDLUInt) {
    if (dtype.bits == 8)
      return "uint8";
    else if (dtype.bits == 16)
      return "uint16";
    else if (dtype.bits == 32)
      return "uint32";
    else if (dtype.bits == 64)
      return "uint64";
  }
  return std::string(NOT_SUPPORT);
}

struct OpCommonAttr {
  std::string optype = "";
  std::string fid = "";
  uint32_t ndim = 0;
  std::string dtype = "float32";

  OpCommonAttr(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors) {
    auto opT = mindspore::predict::EnumNameOpT(opdef.attr_type());
    this->optype = opT;
    MS_ASSERT(opdef.name() != nullptr);
    this->fid = opdef.name()->str();
    if (!tensors.empty()) {
      MS_ASSERT(tensors.front() != nullptr);
      ndim = tensors.front()->ndim;
      dtype = get_dtype(*tensors.front());
    }
  }
};

template <class T>
static void NCHW2NHWC(DLTensor *src) {
  if (src == nullptr) {
    MS_LOGW("input src is nullptr");
    return;
  }
  T *src_data = static_cast<T *>(src->data);
  std::unique_ptr<T[]> tmp(new (std::nothrow)
                             T[src->shape[NCHW_N] * src->shape[NCHW_C] * src->shape[NCHW_H] * src->shape[NCHW_W]]);
  if (tmp == nullptr) {
    MS_LOGW("new tmp buf failed");
    return;
  }
  int N = src->shape[NCHW_N];
  int C = src->shape[NCHW_C];
  int H = src->shape[NCHW_H];
  int W = src->shape[NCHW_W];

  // NCHW -> NHWC
  int k = 0;
  for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
      for (int w = 0; w < W; w++)
        for (int c = 0; c < C; c++) {
          tmp[k++] = src_data[n * C * H * W + c * H * W + h * W + w];
        }

  int sizes = N * C * H * W * sizeof(T);
  errno_t ret = memcpy_s(src_data, sizes, tmp.get(), sizes);
  if (ret != 0) {
    MS_LOGW("memcpy_s failed: %d", ret);
    return;
  }
}

static void transpose_shape(DLTensor *tensor, std::vector<int64_t> axis) {
  if (tensor == nullptr) {
    MS_LOGW("input tensor is nullptr");
    return;
  }
  int ndim = tensor->ndim;
  std::vector<int64_t> origin_shape(tensor->shape, tensor->shape + ndim);

  for (int i = ndim - 1; i >= 0; --i) {
    tensor->shape[i] = origin_shape[axis[i]];
  }
}

static runnerType Pack_NCHW2NHWC(runnerType fun) {
  if (fun == nullptr) {
    MS_LOGE("input fun is nullptr");
    return nullptr;
  }
  auto runner = [fun](const std::vector<DLTensor *> &tensors) -> int {
    if (tensors.back() == nullptr) {
      MS_LOGE("tensors.back() is nullptr");
      return 1;
    }
    transpose_shape(tensors.back(), {0, 3, 1, 2});  // NHWC -> NCHW
    fun(tensors);

    auto output = tensors.back();
    if (output == nullptr) {
      MS_LOGE("tensors.back() after func is nullptr");
      return 1;
    }
    if (output->dtype.bits == 8) {
      NCHW2NHWC<uint8_t>(output);
    } else if (output->dtype.bits == 16) {
      NCHW2NHWC<uint16_t>(output);
    } else if (output->dtype.bits == 32) {
      NCHW2NHWC<uint32_t>(output);
    } else if (output->dtype.bits == 64) {
      NCHW2NHWC<uint64_t>(output);
    } else {
      MS_LOGE("conv NCHW2NHWC output.dtype.bits=%d invalid, only support (8, 16, 32, 64)", output->dtype.bits);
      return 1;
    }

    if (tensors.back() == nullptr) {
      MS_LOGE("tensors.back() is nullptr");
      return 1;
    }
    transpose_shape(tensors.back(), {0, 2, 3, 1});  // NCHW -> NHWC
    return 0;
  };
  return runner;
}

runnerType __attribute__((noinline)) GetKernel_Insert_vector_int32(const std::string &fid,
                                                                   const std::vector<int32_t> &vec) {
  auto f = GetFunction(fid);
  if (f == nullptr) {
    MS_LOGE("GetFunction return nullptr");
    return nullptr;
  }
  auto runner = [f, vec](const std::vector<DLTensor *> &tensors) -> int {
    std::vector<TVMValue> values(vec.size() + tensors.size());
    std::vector<int> codes(values.size());
    tvm::runtime::TVMArgsSetter setter(values.data(), codes.data());
    for (size_t i = 0; i < vec.size(); ++i) {
      setter(i, vec.at(i));
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
      setter(i + vec.size(), tensors.at(i));
    }
    auto passfunc = reinterpret_cast<BackendPackedCFunc>(f);
    return passfunc(values.data(), codes.data(), values.size());
  };
  return runner;
}

runnerType __attribute__((noinline)) GetKernel_Insert_vector_float(const std::string &fid,
                                                                   const std::vector<float> &vec) {
  auto f = GetFunction(fid);
  if (f == nullptr) {
    MS_LOGE("GetFunction return nullptr");
    return nullptr;
  }
  auto runner = [f, vec](const std::vector<DLTensor *> &tensors) -> int {
    std::vector<TVMValue> values(vec.size() + tensors.size());
    std::vector<int> codes(values.size());
    tvm::runtime::TVMArgsSetter setter(values.data(), codes.data());
    for (size_t i = 0; i < vec.size(); ++i) {
      setter(i, vec.at(i));
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
      setter(i + vec.size(), tensors.at(i));
    }
    auto passfunc = reinterpret_cast<BackendPackedCFunc>(f);
    return passfunc(values.data(), codes.data(), values.size());
  };
  return runner;
}

static runnerType GetKernel_Conv(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                 const KernelOption &option) {
  if (tensors.at(0) == nullptr) {
    MS_LOGE("input tensors.at(0) is nullptr");
    return nullptr;
  }
  int n = tensors.at(0)->shape[NCHW_N];
  int ci = tensors.at(0)->shape[NCHW_C];
  int h = tensors.at(0)->shape[NCHW_H];
  int w = tensors.at(0)->shape[NCHW_W];
  std::vector<int32_t> arg_const{n, ci, h, w};
  const OpCommonAttr opAttr(opdef, tensors);

  std::string fid;
  if (opdef.attr_as_Conv2D() != nullptr) {
    auto op = opdef.attr_as_Conv2D();
    fid = std::string(mindspore::predict::EnumNameOpT(opdef.attr_type())) + "_ndim" + std::to_string(opAttr.ndim) +
          "_" + opAttr.dtype + "_k" + std::to_string(op->kernelH()) + "_s" + std::to_string(op->strideH()) + "_p" +
          std::to_string(op->padUp()) + std::to_string(op->padDown()) + std::to_string(op->padLeft()) +
          std::to_string(op->padRight()) + "_d" + std::to_string(op->dilateH()) + "_act" +
          std::to_string(static_cast<int>(op->activationType())) + "_vc" + std::to_string(1) + "_vh" +
          std::to_string(1) + "_vw" + std::to_string(1) + "_hasbias" + std::to_string(op->hasBias());
    if (tensors.at(1) == nullptr) {
      MS_LOGE("input tensors.at(1) is nullptr");
      return nullptr;
    }
    int co = tensors.at(1)->shape[NCHW_N];
    arg_const.push_back(co);
  } else if (opdef.attr_as_DepthwiseConv2D() != nullptr) {
    auto op = opdef.attr_as_DepthwiseConv2D();
    fid = std::string(mindspore::predict::EnumNameOpT(opdef.attr_type())) + "_ndim" + std::to_string(opAttr.ndim) +
          "_" + opAttr.dtype + "_k" + std::to_string(op->kernelH()) + "_s" + std::to_string(op->strideH()) + "_p" +
          std::to_string(op->padUp()) + std::to_string(op->padDown()) + std::to_string(op->padLeft()) +
          std::to_string(op->padRight()) + "_d" + std::to_string(op->dilateH()) + "_act" +
          std::to_string(static_cast<int>(op->activationType())) + "_vc" + std::to_string(1) + "_vh" +
          std::to_string(1) + "_vw" + std::to_string(1) + "_hasbias" + std::to_string(op->hasBias());
    int co = tensors.at(0)->shape[NCHW_C] * op->channelMultiplier();
    arg_const.push_back(co);
  } else if (opdef.attr_as_DeDepthwiseConv2D() != nullptr) {
    auto op = opdef.attr_as_DeDepthwiseConv2D();
    fid = std::string(mindspore::predict::EnumNameOpT(opdef.attr_type())) + "_ndim" + std::to_string(opAttr.ndim) +
          "_" + opAttr.dtype + "_k" + std::to_string(op->kernelH()) + "_s" + std::to_string(op->strideH()) + "_p" +
          std::to_string(op->padUp()) + std::to_string(op->padDown()) + std::to_string(op->padLeft()) +
          std::to_string(op->padRight()) + "_d" + std::to_string(op->dilateH()) + "_act" +
          std::to_string(static_cast<int>(op->activationType())) + "_vc" + std::to_string(1) + "_vh" +
          std::to_string(1) + "_vw" + std::to_string(1) + "_hasbias" + std::to_string(op->hasBias());
    int co = tensors.at(0)->shape[NCHW_C] * op->channelMultiplier();
    arg_const.push_back(co);
  }
  auto fun = GetKernel(fid);
  if (fun == nullptr) {
    MS_LOGE("GetKernel return nullptr");
    return nullptr;
  }

  auto f = GetFunction(fid);
  if (f == nullptr) {
    MS_LOGE("GetFunction return nullptr");
    return nullptr;
  }
  auto runner = [f, arg_const](const std::vector<DLTensor *> &tensors) -> int {
    int ndim[TENSOR_NUM_MAX];
    int64_t *shapes[TENSOR_NUM_MAX];
    int64_t *strides[TENSOR_NUM_MAX];
    store_shape(tensors, ndim, shapes, strides, STORE_MODE);

    std::vector<TVMValue> values(arg_const.size() + tensors.size());
    std::vector<int> codes(values.size());
    tvm::runtime::TVMArgsSetter setter(values.data(), codes.data());
    for (size_t i = 0; i < arg_const.size(); ++i) {
      setter(i, arg_const.at(i));
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
      setter(i + arg_const.size(), tensors.at(i));
    }
    auto passfunc = reinterpret_cast<BackendPackedCFunc>(f);
    passfunc(values.data(), codes.data(), values.size());
    store_shape(tensors, ndim, shapes, strides, RESUME_MODE);
    return 0;
  };
  fun = runner;

  if (opdef.isLastConv()) {
    return Pack_NCHW2NHWC(fun);
  }
  return fun;
}

void update_shape_NC4HW4(const std::vector<DLTensor *> &tensors, int64_t (&shapeA)[TENSOR_NUM_MAX],
                         int64_t (&shapeC)[TENSOR_NUM_MAX]) {
  auto inputA = tensors.front();
  auto output = tensors.back();
  if (inputA == nullptr) {
    MS_LOGW("input tensors.front() is nullptr");
    return;
  }
  if (output == nullptr) {
    MS_LOGW("input tensors.back() is nullptr");
    return;
  }
  shapeA[inputA->ndim] = tile;
  for (int32_t i = 0; i < inputA->ndim; ++i) {
    if (i == 1) {
      shapeA[i] = inputA->shape[i] >> 2;
    } else {
      shapeA[i] = inputA->shape[i];
    }
  }
  {
    inputA->ndim = inputA->ndim + 1;
    inputA->shape = shapeA;
    inputA->strides = nullptr;
  }
  shapeC[output->ndim] = tile;
  for (int32_t i = 0; i < output->ndim; ++i) {
    if (i == 1) {
      shapeC[i] = output->shape[i] >> 2;
    } else {
      shapeC[i] = output->shape[i];
    }
  }
  {
    output->ndim = output->ndim + 1;
    output->shape = shapeC;
    output->strides = nullptr;
  }
}

static runnerType GetKernel_Conv_var(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                     const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto fun = GetKernel(opAttr.fid);
  if (tensors.at(0) == nullptr) {
    MS_LOGE("input tensors.at(0) is nullptr");
    return nullptr;
  }
  std::string fid = opAttr.fid.substr(0, opAttr.fid.find('_'));
  int n = tensors.at(0)->shape[NCHW_N];
  int ci = tensors.at(0)->shape[NCHW_C];
  int h = tensors.at(0)->shape[NCHW_H];
  int w = tensors.at(0)->shape[NCHW_W];
  int co = tensors.at(1)->shape[NCHW_C];
  std::vector<int32_t> arg_const{n, ci >> 2, h, w, co};
  if (fun == nullptr) {
    auto fd = [](int h, std::vector<int> &res) {
      for (int i = 2; i <= h; i += 2) {
        if ((h % i) == 0) res.emplace_back(i);
      }
    };
    int outidx = tensors.size() - 1;
    std::vector<int> vw;
    if (tensors.at(outidx) == nullptr) {
      MS_LOGE("input tensors.at(%d) is nullptr", outidx);
      return nullptr;
    }
    fd(tensors.at(outidx)->shape[NCHW_W], vw);

    auto op = opdef.attr_as_DeConv2D();
    if (op == nullptr) {
      MS_LOGE("opdef.attr_as_DeConv2D() is nullptr");
      return nullptr;
    }
    std::string fids;
    for (auto iter = vw.rbegin(); iter != vw.rend(); iter++) {
      fids = fid + "_ndim" + std::to_string(opAttr.ndim + 1) + "_" + opAttr.dtype + "_k" +
             std::to_string(op->kernelH()) + "_s" + std::to_string(op->strideH()) + "_p" + std::to_string(op->padUp()) +
             std::to_string(op->padDown()) + std::to_string(op->padLeft()) + std::to_string(op->padRight()) + "_d" +
             std::to_string(op->dilateH()) + "_act" + std::to_string(static_cast<int>(op->activationType())) + "_vc" +
             std::to_string(4) + "_vh" + std::to_string(2) + "_vw" + std::to_string(*iter) + "_hasbias" +
             std::to_string(op->hasBias());
      fun = GetKernel(fids);
      if (fun != nullptr) {
        break;
      }
    }
    fid = fids;
    if (fun == nullptr) {
      MS_LOGE("fun is nullptr");
      return nullptr;
    }
    auto f = GetFunction(fid);
    if (f == nullptr) {
      MS_LOGE("GetFunction return nullptr");
      return nullptr;
    }
    auto runner = [f, arg_const](const std::vector<DLTensor *> &tensors) -> int {
      int ndim[TENSOR_NUM_MAX];
      int64_t *shapes[TENSOR_NUM_MAX];
      int64_t *strides[TENSOR_NUM_MAX];
      int64_t shapeA[TENSOR_NUM_MAX];
      int64_t shapeC[TENSOR_NUM_MAX];
      store_shape(tensors, ndim, shapes, strides, STORE_MODE);
      update_shape_NC4HW4(tensors, shapeA, shapeC);

      std::vector<TVMValue> values(arg_const.size() + tensors.size());
      std::vector<int> codes(values.size());
      tvm::runtime::TVMArgsSetter setter(values.data(), codes.data());
      for (size_t i = 0; i < arg_const.size(); ++i) {
        setter(i, arg_const.at(i));
      }
      for (size_t i = 0; i < tensors.size(); ++i) {
        setter(i + arg_const.size(), tensors.at(i));
      }
      auto passfunc = reinterpret_cast<BackendPackedCFunc>(f);
      passfunc(values.data(), codes.data(), values.size());
      store_shape(tensors, ndim, shapes, strides, RESUME_MODE);
      return 0;
    };
    fun = runner;
  }

  if (opdef.isLastConv()) {
    return Pack_NCHW2NHWC(fun);
  }
  return fun;
}

enum reahpeCHW_Mode { FusedCHW, ExpandCHW };

void update_shape_reahpeCHW(const std::vector<DLTensor *> &tensors, reahpeCHW_Mode mode, int64_t (&shape)[4],
                            int64_t (&strides)[4], bool reahpe_output = false) {
  auto input = tensors.front();
  auto output = tensors.back();
  if (input == nullptr) {
    MS_LOGW("input tensors.front() is nullptr");
    return;
  }
  if (output == nullptr) {
    MS_LOGW("input tensors.back() is nullptr");
    return;
  }
  int ndim;
  if (mode == FusedCHW) {
    ndim = 2;
    int64_t CHW = 1;
    for (int32_t i = 1; i < input->ndim; ++i) {
      CHW *= input->shape[i];
    }
    shape[NCHW_N] = input->shape[NCHW_N];
    shape[NCHW_C] = CHW;
    strides[1] = 1;
    strides[0] = CHW;
  } else {
    ndim = 4;
    shape[NCHW_N] = input->shape[NCHW_N];
    shape[NCHW_C] = input->shape[NCHW_C];
    shape[NCHW_H] = 1;
    shape[NCHW_W] = 1;
    strides[3] = 1;
    strides[2] = 1;
    strides[1] = 1;
    strides[0] = input->shape[NCHW_C];
  }

  input->ndim = ndim;
  input->shape = shape;
  input->strides = strides;
  if (reahpe_output) {
    output->ndim = ndim;
    output->shape = shape;
    output->strides = strides;
  }
}

static runnerType Pack_reahpeCHW(const runnerType &fun, const std::vector<DLTensor *> &tensors, reahpeCHW_Mode mode,
                                 bool reahpe_output = false) {
  if (fun == nullptr) {
    MS_LOGE("input fun is nullptr");
    return nullptr;
  }
  if (tensors.front() == nullptr) {
    MS_LOGE("input tensors.front() is nullptr");
    return nullptr;
  }
  if ((tensors.front()->ndim == 2 && mode == FusedCHW) || (tensors.front()->ndim == 4 && mode == ExpandCHW)) {
    return fun;
  }

  auto runner = [fun, mode, reahpe_output](const std::vector<DLTensor *> &tensors) -> int {
    int ndim[TENSOR_NUM_MAX];
    int64_t *shape[TENSOR_NUM_MAX];
    int64_t *strides[TENSOR_NUM_MAX];
    int64_t shape_R[4];
    int64_t strides_R[4];
    store_shape(tensors, ndim, shape, strides, STORE_MODE);
    update_shape_reahpeCHW(tensors, mode, shape_R, strides_R, reahpe_output);
    fun(tensors);
    store_shape(tensors, ndim, shape, strides, RESUME_MODE);
    return 0;
  };
  return runner;
}

static runnerType GetKernel_BatchNorm(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                      const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  std::string fid;
  std::vector<float> epsilon(1, 0.001);
  if (opAttr.optype == "BatchNorm") {
    fid = "TFBatchNorm_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype + "_axis1";
    epsilon.front() = opdef.attr_as_FusedBatchNorm()->epsilon();
    return GetKernel_Insert_vector_float(fid, epsilon);
  } else if (opAttr.optype == "CaffeBatchNorm") {
    fid = "CaffeBatchNorm_ndim4_" + opAttr.dtype + "_axis1";
    epsilon.front() = opdef.attr_as_CaffeBatchNorm()->epsilon();
    auto fun = GetKernel_Insert_vector_float(fid, epsilon);
    if (fun == nullptr) {
      MS_LOGE("GetKernel_Insert_vector_float return nullptr");
      return nullptr;
    }
    bool reahpe_output = true;
    return Pack_reahpeCHW(fun, tensors, ExpandCHW, reahpe_output);
  } else if (opAttr.optype == "BiasAdd") {
    auto op = opdef.attr_as_BiasAdd();
    fid = "TFBiasAdd_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype + "_axis" +
          std::to_string(op->axis()->Get(0));
    return GetKernel(fid);
  } else if (opAttr.optype == "Scale") {
    fid = "CaffeScale_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype + "_axis1";
    return GetKernel(fid);
  }
  return nullptr;
}

void update_shape_flatten(const std::vector<DLTensor *> &tensors, int64_t *shape, int64_t *strides) {
  auto inputA = tensors.back();
  if (inputA == nullptr) {
    MS_LOGW("input tensors.back() is nullptr");
    return;
  }
  for (int32_t i = 0; i < inputA->ndim; ++i) {
    *shape *= inputA->shape[i];
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    tensors[i]->ndim = 1;
    tensors[i]->shape = shape;
    tensors[i]->strides = strides;
  }
}

std::string GetEltwiseMode(const OpCommonAttr &opAttr, const mindspore::predict::OpDef &opdef) {
  auto &optype = opAttr.optype;
  std::string mode = "add";
  if (optype == "Eltwise") {
    auto op_mode = opdef.attr_as_Eltwise()->mode();
    if (mindspore::predict::EltwiseMode_PROD == op_mode) {
      mode = "multiply";
    } else if (mindspore::predict::EltwiseMode_SUM == op_mode) {
      mode = "add";
    } else if (mindspore::predict::EltwiseMode_MAXIMUM == op_mode) {
      mode = "maximum";
    }
  } else {
    if ("Add" == optype) {
      mode = "add";
    } else if ("Sub" == optype) {
      mode = "subtract";
    } else if ("Mul" == optype) {
      mode = "multiply";
    } else if ("RealDiv" == optype) {
      mode = "divide";
    } else if ("Maximum" == optype) {
      mode = "maximum";
    }
  }
  return mode;
}

bool IsSwap(const std::vector<DLTensor *> &tensors) {
  auto CalShape = [](DLTensor *tensor) -> int {
    int res = 1;
    if (tensor == nullptr) {
      MS_LOGE("input DLTensor is nullptr");
      return -1;
    }
    for (int i = 0; i < tensor->ndim; ++i) {
      res *= tensor->shape[i];
    }
    return res;
  };

  MS_ASSERT(tensors[0] != nullptr);
  MS_ASSERT(tensors[1] != nullptr);
  auto ndimA = tensors[0]->ndim;
  auto ndimB = tensors[1]->ndim;
  bool isSwap = false;

  if (ndimA <= ndimB) {
    auto AShape = CalShape(tensors[0]);
    auto BShape = CalShape(tensors[1]);
    if (AShape < BShape) {
      isSwap = true;
    }
  }
  return isSwap;
}

static runnerType GetKernel_Eltwise(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                    const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  std::string mode = GetEltwiseMode(opAttr, opdef);

  // make fid
  int indexA = 0;
  int indexB = 1;
  MS_ASSERT(tensors[0] != nullptr);
  MS_ASSERT(tensors[1] != nullptr);
  auto ndimA = tensors[0]->ndim;
  auto ndimB = tensors[1]->ndim;

  bool isSwap = IsSwap(tensors);
  if (isSwap) {
    std::swap(ndimA, ndimB);
    std::swap(indexA, indexB);
  }

  MS_ASSERT(tensors[indexA] != nullptr);
  MS_ASSERT(tensors[indexB] != nullptr);
  if (ndimA == 1 && tensors[indexA]->shape[NCHW_N] == 1) {
    ndimA = 0;
  }
  if (ndimB == 1 && tensors[indexB]->shape[NCHW_N] == 1) {
    ndimB = 0;
  }
  bool is_same = ndimA == ndimB && ndimA > 1;
  for (int i = 0; i < tensors[indexB]->ndim && is_same; ++i) {
    if (tensors[indexB]->shape[i] != tensors[indexA]->shape[i]) {
      is_same = false;
    }
  }
  for (int i = 0; i < tensors[indexB]->ndim && ndimB > 1 && is_same == false; ++i) {
    if (tensors[indexB]->shape[i] == 1) {
      ndimB--;
    }
  }

  if (ndimA == ndimB && ndimA >= 1) {
    std::string fid = "Eltwise_" + mode + "_ndimA1_ndimB1" + "_" + opAttr.dtype;
    auto fun = GetKernel(fid);
    if (fun == nullptr) {
      MS_LOGE("GetKernel return nullptr");
      return nullptr;
    }
    auto runner = [fun, isSwap](const std::vector<DLTensor *> &tensors) -> int {
      std::vector<DLTensor *> tensorsCopy(tensors);
      if (isSwap) {
        iter_swap(tensorsCopy.begin(), tensorsCopy.begin() + 1);
      }
      int ndim[TENSOR_NUM_MAX];
      int64_t *shapes[TENSOR_NUM_MAX];
      int64_t *strides[TENSOR_NUM_MAX];
      int64_t shape = 1;
      int64_t stride = 1;

      store_shape(tensorsCopy, ndim, shapes, strides, STORE_MODE);
      update_shape_flatten(tensorsCopy, &shape, &stride);
      fun(tensorsCopy);
      store_shape(tensorsCopy, ndim, shapes, strides, RESUME_MODE);
      return 0;
    };
    return runner;
  } else {
    std::string fid =
      "Eltwise_" + mode + "_ndimA" + std::to_string(ndimA) + "_ndimB" + std::to_string(ndimB) + "_" + opAttr.dtype;
    auto fun = GetKernel(fid);
    if (fun == nullptr) {
      MS_LOGE("GetKernel return nullptr");
      return nullptr;
    }
    auto runner = [fun, isSwap](const std::vector<DLTensor *> &tensors) -> int {
      std::vector<DLTensor *> tensorsCopy(tensors);
      if (isSwap) {
        iter_swap(tensorsCopy.begin(), tensorsCopy.begin() + 1);
      }

      fun(tensorsCopy);
      return 0;
    };
    return runner;
  }
}

static runnerType GetKernel_Resize(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                   const KernelOption &option) {
  if (tensors.size() != 2) {
    MS_LOGE("Input tensors num should be 2 !");
    return nullptr;
  }
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_Resize();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_Resize() is nullptr");
    return nullptr;
  }
  std::string fid = "Resize_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype;
  if (op->method() == mindspore::predict::ResizeMethod::ResizeMethod_NEAREST_NEIGHBOR) {
    fid += "_nearest_neighbor";
  } else if (op->method() == mindspore::predict::ResizeMethod::ResizeMethod_BILINEAR) {
    fid += "_bilinear";
  }
  fid += (op->alignCorners()) ? "_Align" : "_NotAlign";
  std::vector<int32_t> HeightWidth = {
    static_cast<int32_t>(op->newHeight()),
    static_cast<int32_t>(op->newWidth()),
  };
  return GetKernel_Insert_vector_int32(fid, HeightWidth);
}

static runnerType GetKernel_DataCarry(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                      const KernelOption &option) {
  auto runner = [](const std::vector<DLTensor *> &tensors) -> int {
    auto input = tensors.front();
    auto output = tensors.back();
    if (input == nullptr) {
      MS_LOGE("input tensors.front() is nullptr");
      return 1;
    }
    if (output == nullptr) {
      MS_LOGE("input tensors.back() is nullptr");
      return 1;
    }
    uint64_t input_num = 1;
    for (int i = 0; i < input->ndim; ++i) {
      input_num *= input->shape[i];
    }
    uint64_t input_byte_num = input_num * input->dtype.lanes * input->dtype.bits / 8;

    uint64_t output_num = 1;
    for (int i = 0; i < output->ndim; ++i) {
      output_num *= output->shape[i];
    }
    uint64_t output_byte_num = output_num * output->dtype.lanes * output->dtype.bits / 8;

    errno_t ret = memcpy_s(output->data, output_byte_num, input->data, input_byte_num);
    if (ret != 0) {
      MS_LOGE("memset_s failed.");
      return ret;
    }
    return 0;
  };
  return runner;
}

static runnerType GetKernel_Shape(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                  const KernelOption &option) {
  auto runner = [](const std::vector<DLTensor *> &tensors) -> int {
    auto input = tensors.front();
    auto output = tensors.back();
    if (input == nullptr) {
      MS_LOGE("input tensors.front() is nullptr");
      return 1;
    }
    if (output == nullptr) {
      MS_LOGE("input tensors.back() is nullptr");
      return 1;
    }
    for (int i = 0; i < input->ndim; ++i) {
      reinterpret_cast<int32_t *>(output->data)[i] = static_cast<int32_t>(input->shape[i]);
    }
    return 0;
  };
  return runner;
}

void StridedSliceArgs(const std::vector<int> &input_shape, std::vector<int> *begin, std::vector<int> *end,
                      std::vector<int> *stride, uint32_t begin_mask, uint32_t end_mask, uint32_t ellipsis_mask,
                      uint32_t new_axis_mask, uint32_t shrink_axis_mask) {
  MS_ASSERT(begin != nullptr);
  MS_ASSERT(end != nullptr);
  MS_ASSERT(stride != nullptr);
  constexpr int support_dims = 8;
  std::bitset<support_dims> begin_list(begin_mask);
  std::bitset<support_dims> end_list(end_mask);
  std::bitset<support_dims> ellipsis_list(ellipsis_mask);
  std::bitset<support_dims> new_axis_list(new_axis_mask);
  std::bitset<support_dims> shrink_list(shrink_axis_mask);

  std::string begin_list_s = begin_list.to_string().substr(support_dims - begin->size());
  reverse(begin_list_s.begin(), begin_list_s.end());

  std::string end_list_s = end_list.to_string().substr(support_dims - end->size());
  reverse(end_list_s.begin(), end_list_s.end());

  std::string ellipsis_list_s = ellipsis_list.to_string().substr(support_dims - end->size());
  reverse(ellipsis_list_s.begin(), ellipsis_list_s.end());

  std::string new_axis_list_s = new_axis_list.to_string().substr(support_dims - end->size());
  reverse(new_axis_list_s.begin(), new_axis_list_s.end());

  std::string shrink_list_s = shrink_list.to_string().substr(support_dims - end->size());
  reverse(shrink_list_s.begin(), shrink_list_s.end());

  int new_axis_count = new_axis_list.count();
  if (ellipsis_list.any()) {
    auto idx = 0;  // ellipsis_list._Find_first();
    // the 1 is ellipsis
    int ellipsis_length = input_shape.size() - (begin->size() - 1 - new_axis_count);
    begin->erase(begin->begin() + idx);
    end->erase(end->begin() + idx);
    stride->erase(stride->begin() + idx);

    begin_list_s.erase(idx, 1);
    end_list_s.erase(idx, 1);
    ellipsis_list_s.erase(idx, 1);
    new_axis_list_s.erase(idx, 1);
    shrink_list_s.erase(idx, 1);

    if (ellipsis_length > 0) {
      begin->insert(begin->begin() + idx, ellipsis_length, 0);
      end->insert(end->begin() + idx, ellipsis_length, 0);
      stride->insert(stride->begin() + idx, ellipsis_length, 1);
      begin_list_s.insert(idx, ellipsis_length, '1');
      end_list_s.insert(idx, ellipsis_length, '1');
      ellipsis_list_s.insert(idx, ellipsis_length, '0');
      new_axis_list_s.insert(idx, ellipsis_length, '0');
      shrink_list_s.insert(idx, ellipsis_length, '0');
    }
  }

  if (new_axis_count) {
    for (int i = static_cast<int>(new_axis_list_s.size()) - 1; i >= 0; i--) {
      if (new_axis_list_s[i] == '1') {
        begin->erase(begin->begin() + i);
        end->erase(end->begin() + i);
        stride->erase(stride->begin() + i);
        begin_list_s.erase(i, 1);
        end_list_s.erase(i, 1);
        shrink_list_s.erase(i, 1);
      }
    }
  }

  unsigned int size = begin->size();
  for (unsigned int i = 0; i < size; i++) {
    if (shrink_list_s[i] == '1') {
      auto beginItr = (begin->begin() + i);
      auto endItr = (end->begin() + i);
      auto strideItr = (stride->begin() + i);
      *endItr = *beginItr + 1;
      *strideItr = 1;
      continue;
    }
    if (begin_list_s[i] == '1') {
      auto beginItr = (begin->begin() + i);
      *beginItr = 0;
    }
    if (end_list_s[i] == '1') {
      auto endItr = (end->begin() + i);
      *endItr = input_shape[i];
    }
  }
}

#define MAXDIMS 10
template <typename T>
int StridedSlice(const std::vector<int> &input_shape, T *input, T *output, int *start, int *end, int *stride,
                 const int &output_size) {
  MS_ASSERT(input != nullptr);
  MS_ASSERT(output != nullptr);
  MS_ASSERT(start != nullptr);
  MS_ASSERT(end != nullptr);
  MS_ASSERT(stride != nullptr);
  int dimension = input_shape.size();
  if (dimension == 1) {
    if (*stride == 1) {
      int sizes = (*end - *start) * sizeof(T);
      errno_t ret = memcpy_s(output, output_size * sizeof(T), input + *start, sizes);
      if (ret != 0) {
        MS_LOGE("memset_s failed: %d", ret);
        return ret;
      }
      return 0;
    }
    for (int j = *start, i = 0; j < *end; j += (*stride), i++) {
      output[i] = input[j];
    }
    return 0;
  }

  // adapt higher dimension
  int dimensionArray[MAXDIMS];
  int factorArray[MAXDIMS];
  int totalElement = 0;

  for (int i = 0; i < dimension; i++) {
    dimensionArray[i] = input_shape[i];
    factorArray[i] = i ? factorArray[i - 1] * dimensionArray[i] : dimensionArray[i];
    totalElement = i ? totalElement * dimensionArray[i] : dimensionArray[i];
  }

  int j = 0;
  for (int k = 0; k < totalElement; k++) {
    bool isValid = true;
    for (int i = 0; i < dimension; i++) {
      int tmp = (k / (totalElement / factorArray[i])) % dimensionArray[i];
      if (tmp < start[i] || tmp >= end[i]) {
        isValid = false;
        break;
      }
      isValid = isValid && ((tmp - start[i]) % stride[i] == 0);
    }
    if (isValid) {
      output[j++] = input[k];
    }
  }

  return 0;
}

static runnerType GetKernel_StridedSlice(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                         const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto ndim = opAttr.ndim;

  auto op = opdef.attr_as_StridedSlice();
  if (op == nullptr) {
    MS_LOGE("op is nullptr");
    return nullptr;
  }
  uint32_t begin_mask = op->beginMask();
  uint32_t end_mask = op->endMask();
  uint32_t ellipsis_mask = op->ellipsisMask();
  uint32_t new_axis_mask = op->newAxisMask();
  uint32_t shrink_axis_mask = op->shrinkAxisMask();
  std::vector<int> begin;
  std::vector<int> end;
  std::vector<int> stride;
  for (uint32_t i = 0; i < ndim; ++i) {
    begin.push_back(op->begin()->Get(i));
    end.push_back(op->end()->Get(i));
    stride.push_back(op->stride()->Get(i));
  }

  auto runner = [begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, begin, end,
                 stride](const std::vector<DLTensor *> &tensors) mutable -> int {
    auto input = tensors.front();
    auto output = tensors.back();
    std::vector<int> input_shape;
    for (int i = 0; i < input->ndim; ++i) {
      input_shape.push_back(input->shape[i]);
    }

    int output_size = 1;
    for (int i = 0; i < output->ndim; ++i) {
      output_size *= output->shape[i];
    }

    StridedSliceArgs(input_shape, &begin, &end, &stride, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                     shrink_axis_mask);

    if (input->dtype.lanes != 1) {
      MS_LOGE("StridedSlice input.dtype.lanes=%d invalid, only support 1", input->dtype.lanes);
      return 1;
    }

    if (input->dtype.bits == 16) {
      StridedSlice(input_shape, reinterpret_cast<uint16_t *>(input->data), reinterpret_cast<uint16_t *>(output->data),
                   begin.data(), end.data(), stride.data(), output_size);
    } else if (input->dtype.bits == 32) {
      StridedSlice(input_shape, reinterpret_cast<uint32_t *>(input->data), reinterpret_cast<uint32_t *>(output->data),
                   begin.data(), end.data(), stride.data(), output_size);
    } else {
      MS_LOGE("StridedSlice input.dtype.bits=%d invalid, only support (16, 32)", input->dtype.bits);
      return 1;
    }
    return 0;
  };
  return runner;
}

template <class T>
static void Permute4d(DLTensor *src, DLTensor *dst, const std::vector<int64_t> &shape,
                      const std::vector<int64_t> &strides) {
  MS_ASSERT(src != nullptr);
  MS_ASSERT(dst != nullptr);
  int64_t N = shape[NCHW_N];
  int64_t C = shape[NCHW_C];
  int64_t H = shape[NCHW_H];
  int64_t W = shape[NCHW_W];
  auto src_data = reinterpret_cast<T *>(src->data);
  auto dst_data = reinterpret_cast<T *>(dst->data);
  int k = 0;
  for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
      for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
          dst_data[k++] = src_data[n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3]];
        }
}

static runnerType GetKernel_CaffeCrop(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                      const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_CaffeCrop();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_CaffeCrop() is nullptr");
    return nullptr;
  }
  std::string fid =
    "CaffeCrop_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype + "_axis" + std::to_string(op->axis());

  std::vector<int32_t> offsets(op->offsets()->size());
  for (size_t i = 0; i < offsets.size(); ++i) {
    offsets[i] = op->offsets()->Get(i);
  }
  return GetKernel_Insert_vector_int32(fid, offsets);
}

static runnerType GetKernel_CaffePReLU(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                       const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_CaffePReLU();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_CaffePReLU() is nullptr");
    return nullptr;
  }
  std::string fid = "CaffePReLU_ndim4_" + opAttr.dtype;
  fid += (op->channelShared()) ? "_channelShared" : "_channelNotShared";
  auto fun = GetKernel(fid);
  if (fun == nullptr) {
    return nullptr;
  }
  bool reahpe_output = true;
  return Pack_reahpeCHW(fun, tensors, ExpandCHW, reahpe_output);
}

static runnerType GetKernel_FullConnection(const mindspore::predict::OpDef &opdef,
                                           const std::vector<DLTensor *> &tensors, const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_FullConnection();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_FullConnection() is nullptr");
    return nullptr;
  }
  std::string fid = "FullConnection_ndimA2_" + opAttr.dtype;
  fid += (op->hasBias()) ? "_hasBias" : "_notHasBias";
  auto fun = GetKernel(fid);
  if (fun == nullptr) {
    return nullptr;
  }
  bool reahpe_output = false;
  return Pack_reahpeCHW(fun, tensors, FusedCHW, reahpe_output);
}

static runnerType GetKernel_Power(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                  const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_Power();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_Power() is nullptr");
    return nullptr;
  }
  std::string fid = "Power_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype;
  std::vector<float> pss;
  pss.push_back(op->power());
  pss.push_back(op->scale());
  pss.push_back(op->shift());
  return GetKernel_Insert_vector_float(fid, pss);
}

static runnerType GetKernel_ArgMax(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                   const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_ArgMax();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_ArgMax() is nullptr");
    return nullptr;
  }
  std::string fid =
    "ArgMax_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype + "_axis" + std::to_string(op->axis());
  fid += (op->keepDims()) ? "_keepDims" : "_notKeepDims";
  fid += "_top1";
  if (tensors.back() == nullptr) {
    MS_LOGE("tensors.back() is nullptr");
    return nullptr;
  }
  fid += "_" + get_dtype(*tensors.back());
  return GetKernel(fid);
}

void update_shape_Concat(const std::vector<DLTensor *> &tensors, int32_t axis, int64_t (&shape)[TENSOR_NUM_MAX][3],
                         int64_t (&strides)[TENSOR_NUM_MAX][3]) {
  int64_t shape_low_dim = 1;
  int64_t shape_high_dim = 1;
  auto output = tensors.back();
  if (output == nullptr) {
    MS_LOGW("tensors.back() is nullptr");
    return;
  }
  for (int32_t i = 0; i < axis; ++i) {
    shape_high_dim *= output->shape[i];
  }
  for (int32_t i = axis + 1; i < output->ndim; ++i) {
    shape_low_dim *= output->shape[i];
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    shape[i][0] = shape_high_dim;
    shape[i][1] = tensors[i]->shape[axis];
    shape[i][2] = shape_low_dim;

    strides[i][2] = 1;
    strides[i][1] = shape[i][2];
    strides[i][0] = shape[i][2] * shape[i][1];

    tensors[i]->ndim = 3;
    tensors[i]->shape = shape[i];
    tensors[i]->strides = strides[i];
  }
}

static runnerType GetKernel_Concat(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                   const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  if (tensors.size() < 2) {
    MS_LOGE("Concat should have at least two tensors");
    return nullptr;
  }
  if (tensors.at(0) == nullptr) {
    MS_LOGE("0th tensors of Concat is nullptr");
    return nullptr;
  }
  auto ndim = tensors.at(0)->ndim;
  if (opdef.attr_as_Concat() == nullptr) {
    MS_LOGE("opdef.attr_as_Concat() is nullptr");
    return nullptr;
  }
  auto axis = opdef.attr_as_Concat()->axis();
  if (axis < 0) {
    axis += ndim;
  }
  std::string fid =
    "Concat_ndim3_" + opAttr.dtype + "_input_num" + std::to_string(static_cast<int>(tensors.size()) - 1) + "_axis1";
  auto fun = GetKernel(fid);
  if (fun == nullptr) {
    MS_LOGE("GetKernel return nullptr");
    return nullptr;
  }
  auto runner = [fun, axis](const std::vector<DLTensor *> &tensors) -> int {
    int ndim[TENSOR_NUM_MAX];
    int64_t *shape[TENSOR_NUM_MAX];
    int64_t *strides[TENSOR_NUM_MAX];
    int64_t shape_C[TENSOR_NUM_MAX][3];
    int64_t strides_C[TENSOR_NUM_MAX][3];
    store_shape(tensors, ndim, shape, strides, STORE_MODE);
    update_shape_Concat(tensors, axis, shape_C, strides_C);
    fun(tensors);
    store_shape(tensors, ndim, shape, strides, RESUME_MODE);
    return 0;
  };
  return runner;
}

template <typename T>
void Stack_ScaleNumber(const std::vector<DLTensor *> &tensors) {
  if (tensors.empty()) {
    MS_LOGW("input tensors is nullptr");
    return;
  }
  auto output = tensors.back();
  if (output != nullptr) {
    MS_LOGW("tensors.back() is nullptr");
    return;
  }
  for (int i = 0; i < static_cast<int>(tensors.size()) - 1; i++) {
    reinterpret_cast<T *>(output->data)[i] = reinterpret_cast<T *>(tensors.at(i)->data)[0];
  }
}

static runnerType GetKernel_Stack(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                  const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_Stack();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_Stack() is nullptr");
    return nullptr;
  }
  if (op->isScale()->Get(0)) {
    auto runner = [](const std::vector<DLTensor *> &tensors) -> int {
      auto input = tensors.front();
      if (input->dtype.bits == 8) {
        Stack_ScaleNumber<uint8_t>(tensors);
      } else if (input->dtype.bits == 16) {
        Stack_ScaleNumber<uint16_t>(tensors);
      } else if (input->dtype.bits == 32) {
        Stack_ScaleNumber<uint32_t>(tensors);
      } else if (input->dtype.bits == 64) {
        Stack_ScaleNumber<uint64_t>(tensors);
      } else {
        MS_LOGE("StridedSlice input.dtype.bits=%d invalid, only support (8, 16, 32, 64)", input->dtype.bits);
        return 1;
      }
      return 0;
    };
    return runner;
  }
  std::string fid = "Stack_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype + "_input_num" +
                    std::to_string(static_cast<int>(tensors.size()) - 1) + "_axis" + std::to_string(op->axis());
  return GetKernel(fid);
}

static runnerType GetKernel_Pad(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_Pad();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_Pad() is nullptr");
    return nullptr;
  }
  std::string fid = "Pad_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype + "_" +
                    mindspore::predict::EnumNamePaddingMode(op->paddingmode());
  std::vector<int32_t> paddings(op->paddings()->size());
  for (size_t i = 0; i < paddings.size(); ++i) {
    paddings[i] = op->paddings()->Get(i);
  }
  return GetKernel_Insert_vector_int32(fid, paddings);
}

static runnerType GetKernel_Pooling(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                    const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_Pooling();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_Pooling() is nullptr");
    return nullptr;
  }
  auto H = tensors.front()->shape[NCHW_H];
  auto W = tensors.front()->shape[NCHW_W];
  auto padUp = op->padUp();
  auto padDown = op->padDown();
  auto padLeft = op->padLeft();
  auto padRight = op->padRight();
  bool useGlobal = false;
  if (H == op->windowH() && W == op->windowW()) {
    useGlobal = true;
  }
  if (op->padMode() != mindspore::predict::PadMode_VALID) {
    int64_t outputHeight = tensors.back()->shape[NCHW_H];
    int64_t outputWidth = tensors.back()->shape[NCHW_W];

    int64_t dHeight = (outputHeight - 1) * op->strideH() + op->windowH() - H;
    int64_t dWidth = (outputWidth - 1) * op->strideW() + op->windowW() - W;
    padUp = dHeight / 2;
    padDown = dHeight - dHeight / 2;
    padLeft = dWidth / 2;
    padRight = dWidth - dWidth / 2;
    if (padDown < 0) {
      padDown = 0;
    }
    if (padRight < 0) {
      padRight = 0;
    }
  }
  std::string poolingMode = mindspore::predict::EnumNamesPoolMode()[op->poolingMode()];
  if (poolingMode != "MAX_POOLING" && poolingMode != "MEAN_POOLING") {
    MS_LOGE("Pooling op not support poolingMode=%s", poolingMode.c_str());
    return nullptr;
  }

  std::string fid;
  fid += useGlobal ? "GlobalPooling" : "Pooling";
  fid += "_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype;
  fid += (poolingMode == "MAX_POOLING") ? "_max" : "_avg";
  if (!useGlobal) {
    fid += "_kernel" + std::to_string(op->windowH()) + std::to_string(op->windowW());
    fid += "_stride" + std::to_string(op->strideH()) + std::to_string(op->strideW());
    fid +=
      "_pad" + std::to_string(padUp) + std::to_string(padDown) + std::to_string(padLeft) + std::to_string(padRight);
    if (op->caffeMode() && (padUp || padDown || padLeft || padRight)) fid += "_caffe";
  }
  auto fun = GetKernel(fid);
  if (fun == nullptr) {
    MS_LOGE("GetKernel return nullptr");
    return nullptr;
  }
  return (opdef.isLastConv()) ? Pack_NCHW2NHWC(fun) : fun;
}

static runnerType GetKernel_Mean(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                 const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_Mean();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_Mean() is nullptr");
    return nullptr;
  }
  std::string axis_str = "";
  for (uint32_t i = 0; i < op->axis()->size(); ++i) {
    axis_str += std::to_string(op->axis()->Get(i));
  }
  std::string fid = "Mean_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype + "_axis" + axis_str;
  fid += (op->keepDims()) ? "_keepDims" : "_notkeepDims";
  return GetKernel(fid);
}

static runnerType GetKernel_MatMul(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                   const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_MatMul();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_MatMul() is nullptr");
    return nullptr;
  }
  std::string fid = "MatMul_ndimA2_ndimB2_" + opAttr.dtype;
  fid += (op->transposeA()) ? "_1" : "_0";
  fid += (op->transposeB()) ? "_1" : "_0";
  return GetKernel(fid);
}

static runnerType GetKernel_Softmax(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                    const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_SoftMax();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_SoftMax() is nullptr");
    return nullptr;
  }
  std::string fid =
    "Softmax_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype + "_axis" + std::to_string(op->axis()->Get(0));
  return GetKernel(fid);
}

void update_shape_Activation(const std::vector<DLTensor *> &tensors, int64_t *shape, int64_t *strides) {
  auto input = tensors.front();
  MS_ASSERT(input != nullptr);
  for (int32_t i = 0; i < input->ndim; ++i) {
    *shape *= input->shape[i];
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    MS_ASSERT(tensors[i] != nullptr);
    tensors[i]->ndim = 1;
    tensors[i]->shape = shape;
    tensors[i]->strides = strides;
  }
}

static runnerType GetKernel_Activation(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                       const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_Activation();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_Activation() is nullptr");
    return nullptr;
  }
  std::string fid =
    "Activation_ndim1_" + opAttr.dtype + "_" + std::string(mindspore::predict::EnumNameActivationType(op->type()));

  auto fun = GetKernel(fid);
  if (fun == nullptr) {
    MS_LOGE("GetKernel return nullptr");
    return nullptr;
  }
  auto runner = [fun](const std::vector<DLTensor *> &tensors) -> int {
    int ndim[TENSOR_NUM_MAX];
    int64_t *shapes[TENSOR_NUM_MAX];
    int64_t *strides[TENSOR_NUM_MAX];
    int64_t shape = 1;
    int64_t stride = 1;

    store_shape(tensors, ndim, shapes, strides, STORE_MODE);
    update_shape_Activation(tensors, &shape, &stride);
    fun(tensors);
    store_shape(tensors, ndim, shapes, strides, RESUME_MODE);
    return 0;
  };
  return runner;
}

static runnerType GetKernel_Exp(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  std::string fid = "Exp_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype;
  return GetKernel(fid);
}

static runnerType GetKernel_Cast(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                 const KernelOption &option) {
  MS_ASSERT(tensors.front() != nullptr);
  MS_ASSERT(tensors.back() != nullptr);
  auto src_dtype = get_dtype(*tensors.front());
  auto dst_dtype = get_dtype(*tensors.back());
  std::string fid = "Cast_ndim" + std::to_string(tensors.front()->ndim) + "_" + src_dtype + "_" + dst_dtype;
  return GetKernel(fid);
}

static runnerType GetKernel_ExpandDims(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                       const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_ExpandDims();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_ExpandDims() is nullptr");
    return nullptr;
  }
  std::string fid =
    "ExpandDims_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype + "_axis" + std::to_string(op->dim());
  return GetKernel(fid);
}

static runnerType GetKernel_Tile(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                 const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_Tile();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_Tile() is nullptr");
    return nullptr;
  }
  std::string fid = "Tile_ndim" + std::to_string(opAttr.ndim) + "_" + opAttr.dtype;
  std::vector<int32_t> multiples;
  for (size_t i = 0; i < op->multiples()->size(); ++i) {
    multiples.push_back(op->multiples()->Get(i));
  }
  return GetKernel_Insert_vector_int32(fid, multiples);
}

static runnerType GetKernel_Range(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                                  const KernelOption &option) {
  const OpCommonAttr opAttr(opdef, tensors);
  auto op = opdef.attr_as_Range();
  if (op == nullptr) {
    MS_LOGE("opdef.attr_as_Range() is nullptr");
    return nullptr;
  }
  std::string fid = "Range_ndim_" + opAttr.dtype;
  std::vector<int32_t> vec = {static_cast<int32_t>(op->start()), static_cast<int32_t>(op->delta())};

  auto f = GetFunction(fid);
  if (f == nullptr) {
    MS_LOGE("GetFunction returu nullptr");
    return nullptr;
  }
  auto runner = [f, vec](const std::vector<DLTensor *> &tensors_origin) -> int {
    // remove 3 input, only remain output
    const std::vector<DLTensor *> tensors = {tensors_origin.back()};
    std::vector<TVMValue> values(vec.size() + tensors.size());
    std::vector<int> codes(values.size());
    tvm::runtime::TVMArgsSetter setter(values.data(), codes.data());
    for (size_t i = 0; i < vec.size(); ++i) {
      setter(i, vec.at(i));
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
      setter(i + vec.size(), tensors.at(i));
    }
    auto passfunc = reinterpret_cast<BackendPackedCFunc>(f);
    return passfunc(values.data(), codes.data(), values.size());
  };
  return runner;
}

using GetKernelFunType = std::function<runnerType(const mindspore::predict::OpDef &opdef,
                                                  const std::vector<DLTensor *> &tensors, const KernelOption &option)>;

static const std::unordered_map<std::string, GetKernelFunType> g_kernel_op_list = {
  {"Conv2D", GetKernel_Conv},
  {"DepthwiseConv2D", GetKernel_Conv},
  {"DeDepthwiseConv2D", GetKernel_Conv},
  {"DeConv2D", GetKernel_Conv_var},
  {"BatchNorm", GetKernel_BatchNorm},
  {"CaffeBatchNorm", GetKernel_BatchNorm},
  {"BiasAdd", GetKernel_BatchNorm},
  {"Scale", GetKernel_BatchNorm},
  {"Eltwise", GetKernel_Eltwise},
  {"Add", GetKernel_Eltwise},
  {"Sub", GetKernel_Eltwise},
  {"Mul", GetKernel_Eltwise},
  {"RealDiv", GetKernel_Eltwise},
  {"Maximum", GetKernel_Eltwise},
  {"ResizeBilinear", GetKernel_Resize},
  {"ResizeNearestNeighbor", GetKernel_Resize},
  {"Squeeze", GetKernel_DataCarry},
  {"Reshape", GetKernel_DataCarry},
  {"Shape", GetKernel_Shape},
  {"StridedSlice", GetKernel_StridedSlice},
  {"CaffeCrop", GetKernel_CaffeCrop},
  {"CaffePReLU", GetKernel_CaffePReLU},
  {"FullConnection", GetKernel_FullConnection},
  {"Power", GetKernel_Power},
  {"ArgMax", GetKernel_ArgMax},
  {"Concat", GetKernel_Concat},
  {"Stack", GetKernel_Stack},
  {"Pad", GetKernel_Pad},
  {"Pooling", GetKernel_Pooling},
  {"Mean", GetKernel_Mean},
  {"MatMul", GetKernel_MatMul},
  {"SoftMax", GetKernel_Softmax},
  {"Activation", GetKernel_Activation},
  {"Exp", GetKernel_Exp},
  {"Cast", GetKernel_Cast},
  {"ExpandDims", GetKernel_ExpandDims},
  {"Tile", GetKernel_Tile},
  {"Range", GetKernel_Range},
};

GetKernelFunType Get_GetKernelFun(const std::string &optype) {
  auto it = g_kernel_op_list.find(optype);
  return (it != g_kernel_op_list.end() ? it->second : nullptr);
}
}  // namespace auto_tensor

runnerType GetKernel(const mindspore::predict::OpDef &opdef, const std::vector<DLTensor *> &tensors,
                     const KernelOption &option) {
  std::string optype = mindspore::predict::EnumNameOpT(opdef.attr_type());
  auto GetKernelFun = auto_tensor::Get_GetKernelFun(optype);
  if (GetKernelFun != nullptr) {
    return GetKernelFun(opdef, tensors, option);
  } else {
    return nullptr;
  }
}

int CallKernel(const std::string &fid, const std::vector<DLTensor *> &tensors) {
  auto runner = GetKernel(fid);
  return runner(tensors);
}

#endif  // LITE_RUNTIME_ON
