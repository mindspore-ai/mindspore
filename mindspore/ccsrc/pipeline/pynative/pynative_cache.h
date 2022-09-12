/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_ABS_CACHE_H
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_ABS_CACHE_H
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <unordered_map>
#include "pybind11/pytypes.h"
#include "utils/hash_map.h"
#include "utils/ms_utils.h"
#include "ir/anf.h"
#include "ir/signature.h"

namespace mindspore {
namespace pynative {
// This map is used to record which input of operator needs to convert to tensor.
// This tensor is passed to grad graph to prevent pass foding optimization.
static std::unordered_map<std::string, std::vector<size_t>> kDynamicInputOpMap = {
  {"AvgPool3DGrad", {0}},
  {"Bernoulli", {1}},
  {"ConjugateTranspose", {1}},
  {"Conv2dBackpropFilter", {2}},
  {"Conv2DTranspose", {2}},
  {"Conv2dBackpropInput", {2}},
  {"Conv3DBackpropFilter", {2}},
  {"Conv3DBackpropInput", {2}},
  {"CropAndResize", {3}},
  {"CTCLossV2", {2, 3}},
  {"CTCLossV2Grad", {2, 3}},
  {"Cumprod", {1}},
  {"CumSum", {1}},
  {"EmbeddingLookup", {2}},
  {"ExpandDims", {1}},
  {"Fill", {1, 2}},
  {"Fills", {1}},
  {"Gather", {2}},
  {"GatherD", {1}},
  {"Greater", {0, 1}},
  {"GreaterEqual", {0, 1}},
  {"IndexFill", {1}},
  {"InvertPermutation", {0}},
  {"Lerp", {2}},
  {"Less", {0, 1}},
  {"LessEqual", {0, 1}},
  {"LinSpace", {2}},
  {"MaskedFill", {2}},
  {"Multinomial", {1}},
  {"NthElement", {1}},
  {"OneHot", {1}},
  {"PadV3Grad", {0}},
  {"Padding", {1}},
  {"ParallelConcat", {0}},
  {"RandomCategorical", {1, 2}},
  {"Poisson", {0}},
  {"ReduceAll", {1}},
  {"ReduceAny", {1}},
  {"ReduceMax", {1}},
  {"ReduceMean", {1}},
  {"ReduceMin", {1}},
  {"ReduceProd", {1}},
  {"ReduceSum", {1}},
  {"Reshape", {1}},
  {"ResizeBilinearV2", {1}},
  {"ScatterNd", {2}},
  {"Slice", {1, 2}},
  {"SliceGrad", {1, 2}},
  {"StandardNormal", {0}},
  {"Tile", {1}},
  {"TopK", {1}},
  {"Transpose", {1}},
  {"TruncateDiv", {0, 1}},
  {"TruncateMod", {0, 1}},
  {"UniformInt", {0}},
  {"UniformReal", {0}},
  {"UnsortedSegmentMax", {2}},
  {"UnsortedSegmentMin", {2}},
  {"UnsortedSegmentProd", {2}},
  {"UnsortedSegmentSum", {2}},
  {"Xdivy", {0, 1}},
  {"Xlogy", {0, 1}},
  {"ScalarToTensor", {0}},
  {"ScalarToArray", {0}},
  {"StandardLaplace", {0}},
  {"UniqueWithPad", {1}},
  {"ApplyAdadelta", {3, 4, 5}},
  {"ApplyAdagrad", {2}},
  {"ApplyAdagradV2", {2}},
  {"ApplyAdaMax", {3, 4, 5, 6, 7}},
  {"ApplyAdamWithAmsgrad", {4, 5, 6}},
  {"ApplyAddSign", {2, 3, 4, 5}},
  {"ApplyCenteredRmsProp", {6, 7, 8}},
  {"ApplyFtrl", {4, 5, 6}},
  {"ApplyGradientDescent", {1}},
  {"ApplyKerasMomentum", {2, 4}},
  {"ApplyMomentum", {2, 4}},
  {"ApplyPowerSign", {2, 3, 4, 5}},
  {"ApplyProximalAdagrad", {2, 3, 4}},
  {"ApplyProximalGradientDescent", {1, 2, 3}},
  {"ApplyRmsProp", {5, 6, 7}},
  {"SparseApplyAdadelta", {3, 4}},
  {"SparseApplyAdagradDA", {5, 6, 7}},
  {"SparseApplyCenteredRMSProp", {4, 5, 6, 7}},
  {"SparseApplyMomentum", {2, 5}},
  {"SparseApplyProximalAdagrad", {2, 3, 4}},
  {"SparseApplyProximalGradientDescent", {1, 2, 3}},
  {"SparseApplyRMSProp", {3}},
  {"SparseTensorDenseAdd", {2}},
  {"SparseTensorDenseMatMul", {2}},
  {"SparseToDense", {3}},
  {"StridedSlice", {2, 3, 4}},
  {"StridedSliceGrad", {2, 3, 4, 5}}};

// The following structures used to get output abstract of op from cache
struct AbsCacheKey {
  std::string prim_name_;
  size_t prim_hash_value_;
  mindspore::HashMap<std::string, ValuePtr> prim_attrs_;
};

struct AbsCacheKeyHasher {
  size_t operator()(const AbsCacheKey &key) const { return key.prim_hash_value_; }
};

struct AbsCacheKeyEqual {
  bool operator()(const AbsCacheKey &lk, const AbsCacheKey &rk) const {
    if (lk.prim_name_ != rk.prim_name_) {
      return false;
    }
    return common::IsAttrsEqual(lk.prim_attrs_, rk.prim_attrs_);
  }
};

struct PrimAbsInfo {
  abstract::AbstractBasePtr abs;
  bool is_dynamic_shape = false;
  mindspore::HashMap<std::string, ValuePtr> attrs;
};
using AbstractListMap = std::unordered_map<abstract::AbstractBasePtrList, PrimAbsInfo,
                                           abstract::AbstractBasePtrListHasher, abstract::AbstractBasePtrListEqual>;
using PrimAbsCache = std::unordered_map<AbsCacheKey, AbstractListMap, AbsCacheKeyHasher, AbsCacheKeyEqual>;

// Used to get input abstract of op from cache
// Key is id of input obj, value is the abstract of input obj
using NodeAbsCache = mindspore::HashMap<std::string, abstract::AbstractBasePtr>;

// Used to cache implicit cast info according to primitive
// Key is primitive name, value is the implicit cast info
struct PrimSignature {
  bool has_dtype_sig;
  std::vector<SignatureEnumDType> dtypes;
  mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> type_indexes;
};
using ImplicitCastCache = mindspore::HashMap<std::string, PrimSignature>;
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_ABS_CACHE_H
