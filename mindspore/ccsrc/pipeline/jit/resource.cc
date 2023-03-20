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

#include "pipeline/jit/resource.h"
#include "ir/dtype.h"
#include "ops/core_ops.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "pipeline/jit/debug/trace.h"
#include "pipeline/jit/parse/data_converter.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/parallel_context.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace pipeline {
BuiltInTypeMap &GetMethodMap() {
  static BuiltInTypeMap method_map = {
    {kObjectTypeString,
     {{"__bool__", std::string("str_bool")},  // C.str_bool
      {"format", std::string("_format")},
      {"__ms_iter__", prim::kPrimIdentity},
      {"lower", prim::kPrimLower}}},
    {kMetaTypeNone,
     {
       {"__bool__", std::string("none_bool")}  // C.none_bool
     }},
    {kObjectTypeFunction,
     {
       {"__bool__", std::string("func_bool")}  // C.str_bool
     }},
    {kNumberTypeBool,
     {
       {"__and__", prim::kPrimBoolAnd},     // P.bool_and
       {"__or__", prim::kPrimBoolOr},       // P.bool_or
       {"__eq__", prim::kPrimBoolEq},       // P.bool_eq
       {"__ne__", std::string("bool_ne")},  // C.bool_ne
       {"__bool__", prim::kPrimIdentity}    // P.identity
     }},
    {kNumberTypeInt,
     {
       {"__add__", prim::kPrimScalarAdd},              // P.scalar_add
       {"__sub__", prim::kPrimScalarSub},              // P.scalar_sub
       {"__mul__", prim::kPrimScalarMul},              // P.scalar_mul
       {"__floordiv__", std::string("int_floordiv")},  // C.int_floordiv
       {"__truediv__", std::string("int_truediv")},    // C.int_truediv
       {"__mod__", prim::kPrimScalarMod},              // P.scalar_mod
       {"__pow__", prim::kPrimScalarPow},              // P.scalar_pow
       {"__floor__", prim::kPrimIdentity},             // P.identity
       {"__trunc__", prim::kPrimIdentity},             // P.identity
       {"__pos__", prim::kPrimScalarUadd},             // P.scalar_uadd
       {"__neg__", prim::kPrimScalarUsub},             // P.scalar_usub
       {"__eq__", prim::kPrimScalarEq},                // P.scalar_eq
       {"__ne__", prim::kPrimScalarNe},                // P.scalar_ne
       {"__lt__", prim::kPrimScalarLt},                // P.scalar_lt
       {"__gt__", prim::kPrimScalarGt},                // P.scalar_gt
       {"__le__", prim::kPrimScalarLe},                // P.scalar_le
       {"__ge__", prim::kPrimScalarGe},                // P.scalar_ge
       {"__bool__", std::string("int_bool")},          // C.int_bool
     }},
    {kNumberTypeUInt,
     {
       {"__add__", prim::kPrimScalarAdd},            // P.scalar_add,
       {"__sub__", prim::kPrimScalarSub},            // P.scalar_sub,
       {"__mul__", prim::kPrimScalarMul},            // P.scalar_mul,
       {"__floordiv__", prim::kPrimScalarDiv},       // P.scalar_div,
       {"__truediv__", std::string("int_truediv")},  // C.int_truediv
       {"__mod__", prim::kPrimScalarMod},            // P.scalar_mod,
       {"__pow__", prim::kPrimScalarPow},            // P.scalar_pow,
       {"__floor__", prim::kPrimIdentity},           // P.identity,
       {"__trunc__", prim::kPrimIdentity},           // P.identity,
       {"__pos__", prim::kPrimScalarUadd},           // P.scalar_uadd,
       {"__neg__", prim::kPrimScalarUsub},           // P.scalar_usub,
       {"__eq__", prim::kPrimScalarEq},              // P.scalar_eq,
       {"__ne__", prim::kPrimScalarNe},              // P.scalar_ne,
       {"__lt__", prim::kPrimScalarLt},              // P.scalar_lt,
       {"__gt__", prim::kPrimScalarGt},              // P.scalar_gt,
       {"__le__", prim::kPrimScalarLe},              // P.scalar_le,
       {"__ge__", prim::kPrimScalarGe},              // P.scalar_ge,
       {"__bool__", std::string("int_bool")},        // C.int_bool
     }},
    {kNumberTypeFloat,
     {
       {"__add__", prim::kPrimScalarAdd},                // P.scalar_add,
       {"__sub__", prim::kPrimScalarSub},                // P.scalar_sub,
       {"__mul__", prim::kPrimScalarMul},                // P.scalar_mul,
       {"__floordiv__", std::string("float_floordiv")},  // C.float_floordiv
       {"__truediv__", prim::kPrimScalarDiv},            // P.scalar_div,
       {"__mod__", prim::kPrimScalarMod},                // P.scalar_mod,
       {"__pow__", prim::kPrimScalarPow},                // P.scalar_pow,
       {"__floor__", prim::kPrimScalarFloor},            // P.scalar_floor,
       {"__trunc__", prim::kPrimScalarTrunc},            // P.scalar_trunc,
       {"__pos__", prim::kPrimScalarUadd},               // P.scalar_uadd,
       {"__neg__", prim::kPrimScalarUsub},               // P.scalar_usub,
       {"__eq__", prim::kPrimScalarEq},                  // P.scalar_eq,
       {"__ne__", prim::kPrimScalarNe},                  // P.scalar_ne,
       {"__lt__", prim::kPrimScalarLt},                  // P.scalar_lt,
       {"__gt__", prim::kPrimScalarGt},                  // P.scalar_gt,
       {"__le__", prim::kPrimScalarLe},                  // P.scalar_le,
       {"__ge__", prim::kPrimScalarGe},                  // P.scalar_ge,
       {"__bool__", std::string("float_bool")},          // C.float_bool
     }},
    {kObjectTypeTuple,
     {
       {"__len__", prim::kPrimSequenceLen},               // P.sequence_len,
       {"__getitem__", prim::kPrimTupleGetItem},          // P.tuple_getitem,
       {"__setitem__", prim::kPrimTupleSetItem},          // P.tuple_setitem,
       {"__ms_iter__", prim::kPrimIdentity},              // P.identity,
       {"__ms_next__", std::string("tuple_next")},        // C.tuple_next,
       {"__ms_hasnext__", std::string("tuple_hasnext")},  // C.tuple_hasnext
       {"__bool__", std::string("tuple_bool")},           // C.tuple_bool
       {"count", prim::kPrimSequenceCount},               // P.sequence_count
       {"index", prim::kPrimSequenceIndex},               // P.sequenc_index
     }},
    {kObjectTypeList,
     {
       {"__len__", prim::kPrimSequenceLen},              // P.sequence_len,
       {"__getitem__", prim::kPrimListGetItem},          // P.list_getitem,
       {"__setitem__", prim::kPrimListSetItem},          // P.list_setitem,
       {"__ms_iter__", prim::kPrimIdentity},             // P.identity
       {"__ms_next__", std::string("list_next")},        // C.list_next
       {"append", std::string("list_append")},           // C.list_append
       {"__bool__", std::string("list_bool")},           // C.list_bool
       {"__ms_hasnext__", std::string("list_hasnext")},  // C.list_hasnext
       {"insert", std::string("list_insert")},           // C.list_insert
       {"pop", std::string("list_pop")},                 // C.list_pop
       {"clear", std::string("list_clear")},             // C.list_clear
       {"reverse", std::string("list_reverse")},         // C.list_reverse
       {"extend", std::string("list_extend")},           // C.list_extend
       {"count", prim::kPrimSequenceCount},              // P.sequence_count
       {"index", prim::kPrimSequenceIndex},              // P.sequence_index
     }},
    {kObjectTypeDictionary,
     {
       {"__len__", prim::kPrimDictLen},                  // P.dict_len
       {"__getitem__", prim::kPrimDictGetItem},          // P.dict_getitem
       {"__setitem__", prim::kPrimDictSetItem},          // P.dict_setitem,
       {"__ms_iter__", prim::kPrimDictGetKeys},          // P.dict_getkeys,
       {"__ms_hasnext__", std::string("dict_hasnext")},  // C.array_hasnext
       {"__ms_next__", std::string("dict_next")},        // C.array_next
       {"keys", prim::kPrimDictGetKeys},                 // P.dict_getkeys,
       {"values", prim::kPrimDictGetValues},             // P.dict_getvalues,
       {"items", prim::kPrimDictItems},                  // P.dict_items
       {"__bool__", std::string("dict_bool")},           // C.dict_bool
       {"get", std::string("dict_get")},                 // C.dict_get
       {"has_key", std::string("dict_haskey")},          // C.dict_haskey
       {"clear", std::string("dict_clear")},             // C.dict_clear
       {"update", std::string("dict_update")},           // C.dict_update
       {"fromkeys", std::string("dict_fromkeys")}        // C.dict_fromkeys
     }},
    {kObjectTypeTensorType,
     {
       {"addcdiv", std::string("addcdiv")},                                // C.addcdiv
       {"addcmul", std::string("addcmul")},                                // C.addcmul
       {"all", std::string("all_")},                                       // C.reduce_all
       {"atan2", std::string("atan2")},                                    // P.Atan2
       {"angle", std::string("angle")},                                    // C.reduce_any
       {"any", std::string("any_")},                                       // C.reduce_any
       {"bincount", std::string("bincount")},                              // bincount
       {"chunk", std::string("chunk")},                                    // chunk
       {"slogdet", std::string("slogdet")},                                // slogdet
       {"trace", std::string("trace")},                                    // trace
       {"tril", std::string("tril")},                                      // tril
       {"__add__", std::string("add")},                                    // C.add
       {"__sub__", std::string("sub")},                                    // C.sub
       {"__mul__", std::string("mul")},                                    // C.mul
       {"__matmul__", std::string("matmul")},                              // F.matmul
       {"xdivy", std::string("xdivy")},                                    // P.Xdivy
       {"abs", std::string("abs_")},                                       // C.abs_
       {"absolute", std::string("abs_")},                                  // C.abs_
       {"mean", std::string("mean")},                                      // C.mean
       {"prod", std::string("prod")},                                      // C.reduce_prod
       {"__truediv__", std::string("truediv")},                            // C.truediv
       {"__floordiv__", std::string("floordiv")},                          // C.floordiv
       {"__mod__", std::string("mod")},                                    // C.mod
       {"__pow__", std::string("pow_")},                                   // C.pow
       {"__floor__", std::string("array_floor")},                          // C.array_floor
       {"__trunc__", std::string("array_trunc")},                          // C.array_trunc
       {"__pos__", std::string("array_uadd")},                             // C.array_uadd
       {"__neg__", std::string("array_usub")},                             // C.array_usub
       {"__eq__", std::string("eq")},                                      // C.eq
       {"__ne__", std::string("ne")},                                      // C.ne
       {"__lt__", std::string("lt")},                                      // C.lt
       {"__gt__", std::string("gt")},                                      // C.gt
       {"__le__", std::string("le")},                                      // C.le
       {"__ge__", std::string("ge")},                                      // C.ge
       {"gt", std::string("gt")},                                          // P.Greater
       {"ge", std::string("ge")},                                          // P.GreaterEqual
       {"expand_as", std::string("expand_tensor_as")},                     // C.expand_as
       {"broadcast_to", std::string("broadcast_to")},                      // P.BroadcastTo
       {"view", std::string("view")},                                      // C.view
       {"view_as", std::string("view_as")},                                // view_as()
       {"__len__", prim::kPrimArrayLen},                                   // P.array_len,
       {"__getitem__", prim::kPrimArrayGetItem},                           // P.array_getitem,
       {"__setitem__", prim::kPrimArraySetItem},                           // P.array_setitem,
       {"__ms_iter__", prim::kPrimIdentity},                               // C.array_iter
       {"__ms_hasnext__", std::string("array_hasnext")},                   // C.array_hasnext
       {"__ms_next__", std::string("array_next")},                         // C.array_next
       {"gather_elements", std::string("gather_elements")},                // P.GatherD
       {"item", std::string("item")},                                      // P.item,
       {"itemset", std::string("itemset")},                                // P.itemset,
       {"transpose", std::string("transpose")},                            // P.transpose
       {"flatten", std::string("flatten")},                                // P.reshape(,-1)
       {"reshape", std::string("reshape")},                                // P.reshape()
       {"reshape_as", std::string("reshape_as")},                          // P.reshape()
       {"reverse", std::string("reverse")},                                // P.ReverseV2()
       {"reverse_sequence", std::string("reverse_sequence")},              // P.ReverseSequence()
       {"bitwise_and", std::string("bitwise_and")},                        // P.BitwiseAnd()
       {"bitwise_or", std::string("bitwise_or")},                          // P.BitwiseOr()
       {"bitwise_xor", std::string("bitwise_xor")},                        // P.BitwiseXor()
       {"bitwise_left_shift", std::string("bitwise_left_shift")},          // bitwise_left_shift
       {"bitwise_right_shift", std::string("bitwise_right_shift")},        // bitwise_right_shift
       {"tan", std::string("tan")},                                        // P.Tan()
       {"ger", std::string("ger")},                                        // P.Ger()
       {"ravel", std::string("ravel")},                                    // P.reshape(,(-1,))
       {"swapaxes", std::string("swapaxes")},                              // P.transpose()
       {"swapdims", std::string("swapdims")},                              // P.transpose()
       {"narrow", std::string("narrow")},                                  // narrow()
       {"masked_fill", std::string("masked_fill")},                        // masked_fill()
       {"masked_select", std::string("masked_select")},                    // masked_select()
       {"nonzero", std::string("nonzero")},                                // nonzero()
       {"expand_dims", std::string("expand_dims")},                        // P.expand_dims()
       {"squeeze", std::string("squeeze")},                                // P.squeeze()
       {"unbind", std::string("unbind")},                                  // P.Unstack()
       {"unsqueeze", std::string("unsqueeze")},                            // P.expand_dims()
       {"astype", std::string("astype")},                                  // P.cast()
       {"short", std::string("short")},                                    // P.cast()
       {"median", std::string("median")},                                  // P.median()
       {"cumsum", std::string("cumsum")},                                  // P.cumsum()
       {"cummin", std::string("cummin")},                                  // cummin()
       {"cummax", std::string("cummax")},                                  // cummax()
       {"index_fill", std::string("index_fill")},                          // index_fill()
       {"index_select", std::string("index_select")},                      // index_select()
       {"repeat_interleave", std::string("repeat_interleave")},            // repeat_interleave()
       {"copy", std::string("copy")},                                      // copy()
       {"copysign", std::string("copysign")},                              // copysign()
       {"inplace_update", std::string("inplace_update")},                  // P.InplaceUpdateV2
       {"lerp", std::string("lerp")},                                      // lerp()
       {"lcm", std::string("lcm")},                                        // F.lcm()
       {"ldexp", std::string("ldexp")},                                    // F.ldexp()
       {"log1p", std::string("log1p")},                                    // P.Log1p()
       {"logit", std::string("logit")},                                    // Logit()
       {"negative", std::string("negative")},                              // neg()
       {"logdet", std::string("logdet")},                                  // logdet()
       {"log_matrix_determinant", std::string("log_matrix_determinant")},  // log_matrix_determinant()
       {"matrix_determinant", std::string("matrix_determinant")},          // matrix_determinant()
       {"matrix_power", std::string("matrix_power")},                      // P.MatrixPower()
       {"det", std::string("det")},                                        // matrix_determinant()
       {"ndimension", std::string("ndim_")},                               // ndimension()
       {"max", std::string("max")},                                        // P.reduce_max()
       {"min", std::string("min")},                                        // P.reduce_min()
       {"pow", std::string("pow")},                                        // P.Pow()
       {"log", std::string("log")},                                        // P.Log()
       {"nelement", std::string("numel")},                                 // numel()
       {"numel", std::string("numel")},                                    // numel()
       {"permute", std::string("permute")},                                // permute()
       {"positive", std::string("positive")},                              // positive()
       {"remainder", std::string("remainder")},                            // remainder()
       {"log10", std::string("log10")},                                    // F.log10()
       {"log2", std::string("log2")},                                      // F.log2()
       {"logaddexp", std::string("logaddexp")},                            // logaddexp()
       {"logaddexp2", std::string("logaddexp2")},                          // logaddexp2()
       {"logsumexp", std::string("logsumexp")},                            // logsumexp()
       {"isneginf", std::string("isneginf")},                              // isneginf()
       {"isposinf", std::string("isposinf")},                              // isposinf()
       {"isreal", std::string("isreal")},                                  // isreal()
       {"minimum", std::string("minimum")},                                // P.Minimum()
       {"cosh", std::string("cosh")},                                      // P.Cosh()
       {"tanh", std::string("tanh")},                                      // P.Tanh()
       {"rad2deg", std::string("rad2deg")},                                // F.rad2deg()
       {"deg2rad", std::string("deg2rad")},                                // F.deg2rad()
       {"round", std::string("round_")},                                   // P.Round()
       {"roll", std::string("roll")},                                      // P.Roll()
       {"rot90", std::string("rot90")},                                    // rot90()
       {"fill", std::string("fill")},                                      // P.fill()
       {"fills", std::string("fills")},                                    // P.fills
       {"ptp", std::string("ptp")},                                        // P.reduce_max() - P.reduce_min()
       {"clamp", std::string("clamp")},                                    // clamp()
       {"clip", std::string("clamp")},                                     // clamp()
       {"__bool__", std::string("tensor_bool")},                           // C.tensor_bool
       {"argmax", std::string("argmax")},                                  // P.Argmax()
       {"argmin", std::string("argmin")},                                  // P.Argmax()
       {"resize", std::string("resize")},                                  // P.Reshape()
       {"crop_and_resize", std::string("crop_and_resize")},                // P.crop_and_resize
       {"select", std::string("select")},                                  // P.Select()
       {"choose", std::string("choose")},                                  // P.Select()
       {"diagonal", std::string("diagonal")},                              // P.Eye()
       {"i0", std::string("i0")},                                          // F.i0()
       {"isclose", std::string("isclose")},                                // P.IsClose()
       {"is_floating_point", std::string("is_floating_point")},            // is_floating_point()
       {"is_signed", std::string("is_signed")},                            // is_signed()
       {"is_complex", std::string("is_complex")},                          // F.is_complex()
       {"inv", std::string("inv")},                                        // inv()
       {"inverse", std::string("inverse")},                                // inverse()
       {"invert", std::string("invert")},                                  // invert()
       {"searchsorted", std::string("searchsorted")},                      // P.Select()
       {"take", std::string("take")},                                      // P.GatherNd()
       {"gather", std::string("gather")},                                  // P.Gather()
       {"scatter", std::string("scatter")},                                // P.TensorScatterElements()
       {"scatter_add", std::string("tensor_scatter_add")},                 // P.TensorScatterAdd()
       {"scatter_mul", std::string("tensor_scatter_mul")},                 // tensor_scatter_mul()
       {"scatter_sub", std::string("tensor_scatter_sub")},                 // P.TensorScatterSub()
       {"scatter_min", std::string("tensor_scatter_min")},                 // P.TensorScatterMin()
       {"scatter_max", std::string("tensor_scatter_max")},                 // P.TensorScatterMax()
       {"scatter_div", std::string("tensor_scatter_div")},                 // P.TensorScatterDiv()
       {"norm", std::string("norm")},                                      // norm()
       {"unsorted_segment_min", std::string("unsorted_segment_min")},      // P.UnsortedSegmentMin()
       {"unsorted_segment_max", std::string("unsorted_segment_max")},      // P.UnsortedSegmentMax()
       {"unsorted_segment_prod", std::string("unsorted_segment_prod")},    // P.UnsortedSegmentProd()
       {"renorm", std::string("renorm")},                                  // renorm()
       {"real", std::string("real")},                                      // real()
       {"reciprocal", std::string("reciprocal")},                          // reciprocal()
       {"rsqrt", std::string("rsqrt")},                                    // rsqrt()
       {"trace", std::string("trace")},                                    // P.Eye()
       {"var", std::string("var")},                                        // P.ReduceSum
       {"std", std::string("std")},                                        // P.ReduceSum
       {"sum", std::string("sum")},                                        // P.ReduceSum
       {"sqrt", std::string("sqrt")},                                      // P.Sqrt()
       {"square", std::string("square")},                                  // P.Square()
       {"sub", std::string("sub")},                                        // P.Sub()
       {"true_divide", std::string("true_divide")},                        // true_divide()
       {"triu", std::string("triu")},                                      // triu()
       {"subtract", std::string("subtract")},                              // true_divide()
       {"sum_to_size", std::string("sum_to_size")},                        // sum_to_size()
       {"exp", std::string("exp")},                                        // P.Exp()
       {"repeat", std::string("repeat")},                                  // C.repeat_elements
       {"bernoulli", prim::kPrimBernoulli},                                // P.Bernoulli()
       {"ceil", std::string("ceil")},                                      // P.Ceil
       {"floor", std::string("floor")},                                    // P.floor
       {"flip", std::string("flip")},                                      // flip
       {"fliplr", std::string("fliplr")},                                  // fliplr
       {"flipud", std::string("flipud")},                                  // flipud
       {"float_power", std::string("float_power")},                        // F.float_power
       {"fmod", std::string("fmod")},                                      // F.fmod
       {"hardshrink", std::string("hardshrink")},                          // P.hshrink
       {"heaviside", std::string("heaviside")},                            // F.heaviside
       {"hypot", std::string("hypot")},                                    // F.hypot
       {"gather_nd", std::string("gather_nd")},                            // P.GatherNd()
       {"unique_consecutive", std::string("unique_consecutive")},          // UniqueConsecutive()
       {"unique_with_pad", std::string("unique_with_pad")},                // P.UniqueWithPad()
       {"diag", std::string("diag")},                                      // P.Diag()
       {"diagflat", std::string("diagflat")},                              // diagflat()
       {"digamma", std::string("digamma")},                                // digamma()
       {"lgamma", std::string("lgamma")},                                  // lgamma()
       {"adaptive_max_pool2d", std::string("adaptive_max_pool2d")},        // P.AdaptiveMaxPool2D
       {"to_coo", std::string("to_coo")},                                  // dense_to_sparse_coo()
       {"to_csr", std::string("to_csr")},                                  // dense_to_sparse_csr()
       {"col2im", std::string("col2im")},                                  // P.Col2Im
       {"split", std::string("split")},                                    // split
       {"tensor_split", std::string("tensor_split")},                      // tensor_split
       {"vsplit", std::string("vsplit")},                                  // vsplit
       {"hsplit", std::string("hsplit")},                                  // hsplit
       {"dsplit", std::string("dsplit")},                                  // dplit
       {"random_categorical", std::string("random_categorical")},          // P.RandomCategorical
       {"xlogy", std::string("xlogy")},                                    // P.Xlogy()
       {"erf", std::string("erf")},                                        // P.Erf()
       {"erfc", std::string("erfc")},                                      // P.Erfc()
       {"argmax_with_value", std::string("argmax_with_value")},            // P.ArgMaxWithValue
       {"argmin_with_value", std::string("argmin_with_value")},            // P.ArgMinWithValue
       {"tile", std::string("tile")},                                      // P.Tile
       {"topk", std::string("topk")},                                      // P.TopK()
       {"isfinite", std::string("isfinite")},                              // P.isfinite()
       {"cos", std::string("cos")},                                        // cos()
       {"cov", std::string("cov")},                                        // cov()
       {"acos", std::string("acos")},                                      // acos()
       {"arccos", std::string("acos")},                                    // acos()
       {"acosh", std::string("acosh")},                                    // acosh()
       {"sigmoid", std::string("sigmoid")},                                // P.Sigmoid()
       {"addr", std::string("addr")},                                      // addr()
       {"add", std::string("add")},                                        // P.Add()
       {"addbmm", std::string("addbmm")},                                  // addbmm()
       {"addmm", std::string("addmm")},                                    // addmm()
       {"addmv", std::string("addmv")},                                    // addmv()
       {"adjoint", std::string("adjoint")},                                // adjoint()
       {"t", std::string("t")},                                            // t()
       {"arccosh", std::string("acosh")},                                  // arccosh()
       {"sin", std::string("sin")},                                        // sin()
       {"sinc", std::string("sinc")},                                      // sinc()
       {"arcsin", std::string("asin")},                                    // arcsin()
       {"arctan", std::string("atan")},                                    // arctan()
       {"arctan2", std::string("atan2")},                                  // arctan2()
       {"asin", std::string("asin")},                                      // asin()
       {"asinh", std::string("asinh")},                                    // asinh()
       {"arcsinh", std::string("asinh")},                                  // arcsinh()
       {"atan", std::string("atan")},                                      // atan()
       {"atanh", std::string("atanh")},                                    // atanh()
       {"arctanh", std::string("atanh")},                                  // arctanh()
       {"baddbmm", std::string("baddbmm")},                                // baddbmm
       {"bmm", std::string("bmm")},                                        // bmm()
       {"value", std::string("value_")},                                   // P.Load(param, U)
       {"to", std::string("to")},                                          // to()
       {"bool", std::string("to_bool")},                                   // bool()
       {"float", std::string("to_float")},                                 // float()
       {"half", std::string("to_half")},                                   // half()
       {"int", std::string("to_int")},                                     // int()
       {"long", std::string("to_long")},                                   // long()
       {"cholesky", std::string("cholesky")},                              // cholesky()
       {"cholesky_inverse", std::string("cholesky_inverse")},              // cholesky_inverse()
       {"conj", std::string("conj")},                                      // conj()
       {"cross", std::string("cross")},                                    // cross()
       {"erfinv", std::string("erfinv")},                                  // erfinv()
       {"less_equal", std::string("less_equal")},                          // less_equal()
       {"fold", std::string("fold")},                                      // fold()
       {"unfold", std::string("unfold")},                                  // unfold()
       {"expand", std::string("expand")},                                  // expand()
       {"cumprod", std::string("cumprod")},                                // cumprod()
       {"div", std::string("div")},                                        // div()
       {"divide", std::string("div")},                                     // divide()
       {"equal", std::string("equal")},                                    // equal()
       {"expm1", std::string("expm1")},                                    // expm1()
       {"eig", std::string("eig")},                                        // eig()
       {"geqrf", std::string("geqrf")},                                    // geqrf()
       {"histc", std::string("histc")},                                    // histc()
       {"dim", prim::kPrimRank},                                           // P.Rank()
       {"index_add", std::string("index_add")},                            // index_add()
       {"greater", std::string("greater")},                                // greater()
       {"greater_equal", std::string("greater_equal")},                    // greater_equal()
       {"igamma", std::string("igamma")},                                  // igamma()
       {"igammac", std::string("igammac")},                                // igammac()
       {"isinf", std::string("isinf")},                                    // isinf()
       {"isnan", std::string("isnan")},                                    // isnan()
       {"le", std::string("le")},                                          // le()
       {"less", std::string("less")},                                      // less()
       {"lt", std::string("less")},                                        // lt()
       {"logical_and", std::string("logical_and")},                        // logical_and()
       {"logical_not", std::string("logical_not")},                        // logical_not()
       {"logical_or", std::string("logical_or")},                          // logical_or()
       {"logical_xor", std::string("logical_xor")},                        // logical_xor()
       {"lstsq", std::string("lstsq")},                                    // lstsq()
       {"mvlgamma", std::string("mvlgamma")},                              // mvlgamma()
       {"matmul", std::string("matmul")},                                  // matmul()
       {"inner", std::string("inner")},                                    // inner()
       {"maximum", std::string("maximum")},                                // maximum()
       {"msort", std::string("msort")},                                    // msort()
       {"mm", std::string("mm")},                                          // mm()
       {"mul", std::string("mul")},                                        // mul()
       {"multiply", std::string("multiply")},                              // multiply()
       {"nan_to_num", std::string("nan_to_num")},                          // nan_to_num()
       {"nansum", std::string("nansum")},                                  // nansum()
       {"neg", std::string("neg")},                                        // neg()
       {"ne", std::string("ne")},                                          // ne()
       {"not_equal", std::string("not_equal")},                            // not_equal()
       {"new_zeros", std::string("new_zeros")},                            // new_zeros()
       {"new_ones", std::string("new_ones")},                              // new_ones()
       {"sgn", std::string("sgn")},                                        // sgn()
       {"sign", std::string("sign")},                                      // sign()
       {"signbit", std::string("signbit")},                                // signbit()
       {"sinh", std::string("sinh")},                                      // sinh()
       {"sort", std::string("sort")},                                      // sort()
       {"argsort", std::string("argsort")},                                // argsort()
       {"trunc", std::string("trunc")},                                    // trunc()
       {"where", std::string("where")},                                    // where()
       {"imag", std::string("imag")},                                      // imag()
       {"diff", std::string("diff")},                                      // diff()
       {"frac", std::string("frac")},                                      // frac()
       {"argwhere", std::string("argwhere")},                              // argwhere()
       {"moveaxis", std::string("moveaxis")},                              // moveaxis()
       {"multinomial", std::string("multinomial")},                        // multinomial()
       {"movedim", std::string("movedim")},                                // movedim()
       {"nextafter", std::string("nextafter")},                            // nextafter()
       {"qr", std::string("qr")},                                          // qr()
     }},
    {kObjectTypeRowTensorType,
     {
       {"__add__", prim::kPrimRowTensorAdd},  // P.row_tensor_add
     }},
    {kObjectTypeCSRTensorType,
     {
       {"astype", std::string("csr_astype")},      // C.csr_astype
       {"abs", std::string("csr_abs")},            // C.csr_abs
       {"sum", std::string("csr_sum")},            // C.csr_sum
       {"mv", std::string("csr_mv")},              // C.csr_mv
       {"to_tuple", std::string("csr_to_tuple")},  // C.csr_to_tuple
       {"to_coo", std::string("csr_to_coo")},      // C.csr_to_coo
       {"to_dense", std::string("csr_to_dense")},  // C.csr_to_dense
       {"mm", std::string("csr_mm")},              // C.csr_mm
       {"add", std::string("csr_add")},            // C.csr_add
       {"softmax", std::string("csr_softmax")},    // C.csr_softmax
     }},
    {kObjectTypeCOOTensorType,
     {
       {"astype", std::string("coo_astype")},      // C.coo_astype
       {"abs", std::string("coo_abs")},            // C.coo_abs
       {"to_tuple", std::string("coo_to_tuple")},  // C.coo_to_tuple
       {"to_csr", std::string("coo_to_csr")},      // C.coo_to_csr
       {"to_dense", std::string("coo_to_dense")},  // C.coo_to_dense
       {"coalesce", std::string("coo_coalesce")},  // C.coo_coalesce
       {"add", std::string("coo_add")},            // C.coo_add
     }},
    {kObjectTypeMapTensorType,
     {
       {"get", std::string("map_tensor_get")},                // C.map_tensor_get
       {"put", std::string("map_tensor_put")},                // C.map_tensor_put
       {"erase", std::string("map_tensor_erase")},            // C.map_tensor_erase
       {"get_keys", std::string("map_tensor_get_keys")},      // C.map_tensor_get_keys
       {"get_values", std::string("map_tensor_get_values")},  // C.map_tensor_get_values
       {"get_data", std::string("map_tensor_get_data")},      // C.map_tensor_get_data
     }},
    {kObjectTypeJTagged, {}},
    {kObjectTypeSymbolicKeyType, {}},
    {kObjectTypeEnvType, {}}};
  return method_map;
}

BuiltInTypeMap &GetAttrMap() {
  static BuiltInTypeMap attr_map = {
    {kObjectTypeTensorType,
     {
       {"shape", prim::kPrimShape},             // C.shape_
       {"dtype", prim::kPrimDType},             // C.dtype_
       {"size", std::string("size_")},          // C.size_
       {"ndim", std::string("ndim_")},          // C.ndim_
       {"H", std::string("H")},                 // C.H
       {"T", std::string("T_")},                // C.T_
       {"itemsize", std::string("itemsize_")},  // C.itemsize_
       {"nbytes", std::string("nbytes_")},      // C.nbytes_
       {"strides", std::string("strides_")},    // C.strides_
       {"mH", std::string("adjoint")},          // C.adjoint
       {"mT", std::string("mT")},               // C.mT_
     }},
    {kObjectTypeRowTensorType,
     {
       {"values", prim::kPrimRowTensorGetValues},           // F.row_tensor_get_values
       {"indices", prim::kPrimRowTensorGetIndices},         // F.row_tensor_get_indices
       {"dense_shape", prim::kPrimRowTensorGetDenseShape},  // F.row_tensor_get_dense_shape
     }},
    {kObjectTypeCOOTensorType,
     {
       {"values", prim::kPrimCOOTensorGetValues},     // F.coo_tensor_get_values
       {"indices", prim::kPrimCOOTensorGetIndices},   // F.coo_tensor_get_indices
       {"shape", prim::kPrimCOOTensorGetDenseShape},  // F.coo_tensor_get_dense_shape
       {"dtype", std::string("dtype_")},              // C.dtype_
       {"size", std::string("sparse_size_")},         // C.sparse_size_
       {"ndim", std::string("sparse_ndim_")},         // C.sparse_ndim_
       {"itemsize", std::string("itemsize_")},        // C.itemsize_
     }},
    {kObjectTypeCSRTensorType,
     {
       {"indptr", prim::kPrimCSRTensorGetIndptr},     // F.csr_tensor_get_indptr
       {"values", prim::kPrimCSRTensorGetValues},     // F.csr_tensor_get_values
       {"indices", prim::kPrimCSRTensorGetIndices},   // F.csr_tensor_get_indices
       {"shape", prim::kPrimCSRTensorGetDenseShape},  // F.csr_tensor_get_shape
       {"dtype", std::string("dtype_")},              // C.dtype_
       {"size", std::string("sparse_size_")},         // C.sparse_size_
       {"ndim", std::string("sparse_ndim_")},         // C.sparse_ndim_
       {"itemsize", std::string("itemsize_")},        // C.itemsize_
     }},
    {kObjectTypeMapTensorType,
     {
       {"default_value", prim::kPrimMapTensorGetDefaultValue},             // F.map_tensor_get_default_value
       {"permit_filter_value", prim::kPrimMapTensorGetPermitFilterValue},  // F.map_tensor_get_permit_filter_value
       {"evict_filter_value", prim::kPrimMapTensorGetEvictFilterValue},    // F.map_tensor_get_evict_filter_value
     }},
  };
  return attr_map;
}

std::mutex Resource::backend_init_mutex_;

Resource::Resource(const py::object &obj)
    : engine_(std::make_shared<abstract::AnalysisEngine>(abstract::GetPrimEvaluatorConstructors(), manager_)),
      source_input_(obj),
      is_cleaned_(false) {}

Resource::~Resource() {
  MS_LOG(DEBUG) << "Resource clear";

  try {
    mindspore::HashMap<std::string, Any>().swap(results_);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Exception when cleaning resource. Error info " << e.what();
  }

  // If exit normally, these global variables will be cleaned
  // in Resource::Clean call by MsPipeline::Compile, but if exit with MS_LOGEXCEPTION,
  // these global variables may not being cleaned, it may
  // cause segmentfault when free python object inside these global variables
  // after python interpreter got freed, so these global variables
  // are cleaned here.
  // So if exit normally, these global variable will be cleaned twice,
  // care be taken to prevent double free in the following functions.
  if (!is_cleaned_) {
    try {
      Clean();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Exception when cleaning resource. Error info " << e.what();
    } catch (...) {
      MS_LOG(ERROR) << "Exception when cleaning resource.";
    }
  }
}

Any GetMethodOrAttr(const string &name, const TypeId &type_id, const BuiltInTypeMap &method_map) {
  auto type_method_map = method_map.find(static_cast<int64_t>(type_id));
  if (type_method_map == method_map.end()) {
    return Any();
  }
  auto method = type_method_map->second.find(name);
  if (method == type_method_map->second.end()) {
    return Any();
  }
  return method->second;
}

bool Resource::IsTypeInBuiltInMap(const TypeId &type) {
  TypeId type_id = NormalizeTypeId(type);
  const BuiltInTypeMap &method_map = GetMethodMap();
  auto iter = method_map.find(static_cast<int64_t>(type_id));
  if (iter == method_map.end()) {
    const BuiltInTypeMap &attr_map = GetAttrMap();
    iter = attr_map.find(static_cast<int64_t>(type_id));
    if (iter == attr_map.end()) {
      return false;
    }
  }
  return true;
}

Any Resource::GetMethodPtr(const TypeId &type, const std::string &name) {
  TypeId type_id = NormalizeTypeId(type);
  const BuiltInTypeMap &method_map = GetMethodMap();
  return GetMethodOrAttr(name, type_id, method_map);
}

Any Resource::GetAttrPtr(const TypeId &type, const std::string &name) {
  TypeId type_id = NormalizeTypeId(type);
  const BuiltInTypeMap &attr_map = GetAttrMap();
  return GetMethodOrAttr(name, type_id, attr_map);
}

void Resource::GetCompileCacheResource(const py::list &compile_cache_dep_files, const py::dict &weights,
                                       const std::string &queue_name, size_t compile_cache_id,
                                       bool *compile_cache_consistent) {
  compile_cache_manager_ = std::make_shared<CompileCacheManager>(compile_cache_id);
  compile_cache_manager_->InitParallelGroupCkptSaveFile();
  MS_EXCEPTION_IF_NULL(compile_cache_consistent);
  if (!*compile_cache_consistent) {
    MS_LOG(WARNING) << "Check the consistency of dependency files hash failed. Execute all the compilation actions.";
    return;
  }
  compile_cache_manager_->InitCompileCacheHash(compile_cache_dep_files);
  *compile_cache_consistent = compile_cache_manager_->CheckDepFilesHashConsistency();
  if (!*compile_cache_consistent) {
    MS_LOG(WARNING) << "Check the consistency of dependency files hash failed. Execute all the compilation actions.";
    return;
  }
  func_graph_ = compile_cache_manager_->GetCachedFuncGraph(manager_, weights, queue_name);
  layout_map_ = compile_cache_manager_->layout_map();
}

void Resource::CacheFuncGraph() const {
  FuncGraphPtr layout_fg = nullptr;
  if (parallel::IsAutoParallelCareGraph(func_graph_)) {
    layout_fg = GetResult(kStepParallelGraph).cast<FuncGraphPtr>();
  }
  compile_cache_manager_->CacheFuncGraph(func_graph_, layout_fg);
}

void Resource::Clean() {
  // Ensure that async backend creating task is finished before clean resource.
  if (backend_ == nullptr && backend_future_.valid()) {
    backend_ = backend_future_.get();
  }
  // AbstractTensor->elements() will be saved in AbstractBasePtrList
  args_abs_.clear();
  source_input_ = py::none();
  // Context with AbstractBasePtrList may be saved in GraphEvaluator
  // some Evaluator like ResolveEvaluator may save Python object in cache,
  // it should be cleaned before Python Interpreter destructed.
  MS_EXCEPTION_IF_NULL(engine_);
  engine_->ClearEvaluatorCache();
  engine_->Clear();
  // Clean cache used for parse. As static variable is released after
  // Python threads is released.
  parse::data_converter::ClearObjectCache();
  parse::Parser::CleanParserResource();
  // Clear all graphs' holding for python object(such as Cell),
  // otherwise it will result to circular reference between the func_graph and cell.
  for (auto graph : manager()->func_graphs()) {
    graph->set_python_obj(nullptr);
  }
  trace::ClearTraceStack();
  is_cleaned_ = true;
}

compile::BackendPtr Resource::GetBackend() const {
  if (backend_ == nullptr && backend_future_.valid()) {
    backend_ = backend_future_.get();
  }
  return backend_;
}

void Resource::SetBackendAsync(std::function<compile::BackendPtr()> func) {
  static const bool is_enable_async = (common::GetEnv("MS_DEV_ASYNC_BACKEND_INIT") == "1");
  static const bool is_enable_ge = (common::GetEnv("MS_ENABLE_GE") == "1");
  if (!is_enable_async || is_enable_ge) {
    // Disable async backend init if required.
    std::lock_guard<std::mutex> guard(GetBackendInitMutex());
    backend_ = func();
    return;
  }
  if (backend_ == nullptr && backend_future_.valid()) {
    (void)backend_future_.get();
  }
  backend_ = nullptr;
  backend_future_ = std::async(std::launch::async, [func]() {
    std::lock_guard<std::mutex> guard(Resource::GetBackendInitMutex());
    return func();
  });
}
}  // namespace pipeline
}  // namespace mindspore
