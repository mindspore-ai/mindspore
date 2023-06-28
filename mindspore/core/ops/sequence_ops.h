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

#ifndef MINDSPORE_CORE_BASE_SEQUENCE_OPS_H_
#define MINDSPORE_CORE_BASE_SEQUENCE_OPS_H_

#include <memory>
#include <string>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "utils/flags.h"

namespace mindspore {
namespace prim {
// Tuple
constexpr auto kRealMakeTuple = "RealMakeTuple";
constexpr auto kMakeTuple = "MakeTuple";
constexpr auto kTupleGetItem = "TupleGetItem";
constexpr auto kTupleSetItem = "tuple_setitem";
constexpr auto kTupleLt = "tuple_lt";
constexpr auto kTupleLe = "tuple_le";
constexpr auto kRealTupleGetItem = "RealTupleGetItem";
constexpr auto kTupleGreaterThan = "tuple_greater_than";
constexpr auto kTupleGreaterEqual = "tuple_greater_equal";
constexpr auto kTupleEqual = "tuple_equal";

// List
constexpr auto kListInplaceClear = "ListInplaceClear";
constexpr auto kListInplaceReverse = "ListInplaceReverse";
constexpr auto kListInplaceExtend = "ListInplaceExtend";
constexpr auto kListInplaceInsert = "ListInplaceInsert";
constexpr auto kListInplacePop = "ListInplacePop";
constexpr auto kMakeList = "MakeList";
constexpr auto kMakeListNew = "make_list";
constexpr auto kListGetItem = "list_getitem";
constexpr auto kListSetItem = "list_setitem";
constexpr auto kListLt = "list_lt";
constexpr auto kListLe = "list_le";
constexpr auto kListGreaterThan = "list_greater_than";
constexpr auto kListGreaterEqual = "list_greater_equal";
constexpr auto kListEqual = "list_equal";
constexpr auto kListDiff = "ListDiff";

// Sequence and Tensor
constexpr auto kTupleToTensor = "TupleToTensor";
constexpr auto kTensorToTuple = "TensorToTuple";
constexpr auto kListToTensor = "ListToTensor";
constexpr auto kTensorToList = "TensorToList";

// Sequence operation
constexpr auto kListAppend = "ListAppend";
constexpr auto kListInsert = "ListInsert";
constexpr auto kListInplaceAppend = "ListInplaceAppend";
constexpr auto kListAppendAndInsertGrad = "ListAppendAndInsertGrad";
constexpr auto kSequenceAdd = "SequenceAdd";
constexpr auto kSequenceCount = "SequenceCount";
constexpr auto kSequenceIndex = "SequenceIndex";
constexpr auto kSequenceMul = "SequenceMul";
constexpr auto kSequenceSlice = "SequenceSlice";
constexpr auto kSequenceLen = "sequence_len";
constexpr auto kSequenceZerosLike = "SequenceZerosLike";
constexpr auto kMakeRange = "make_range";
constexpr auto kSequenceAddOffset = "SequenceAddOffset";
constexpr auto kSequenceSliceGrad = "SequenceSliceGrad";
constexpr auto kSequenceSliceSetItem = "SequenceSliceSetItem";
constexpr auto kSequenceMax = "SequenceMax";
constexpr auto kSequenceMin = "SequenceMin";
constexpr auto kInSequence = "InSequence";
constexpr auto kSequenceAddN = "SequenceAddN";
constexpr auto kSequenceConcat = "SequenceConcat";
constexpr auto kSequenceStack = "SequenceStack";

GVAR_DEF(PrimitivePtr, kPrimSequenceLen, std::make_shared<Primitive>("sequence_len"));

// Array
GVAR_DEF(PrimitivePtr, kPrimArrayLen, std::make_shared<Primitive>("array_len"));
GVAR_DEF(PrimitivePtr, kPrimArrayGetItem, std::make_shared<Primitive>("array_getitem"));
GVAR_DEF(PrimitivePtr, kPrimArraySetItem, std::make_shared<Primitive>("array_setitem"));

// Real tuple and list ops.
GVAR_DEF(PrimitivePtr, kPrimTupleToTensor, std::make_shared<Primitive>(kTupleToTensor));
GVAR_DEF(PrimitivePtr, kPrimRealMakeTuple, std::make_shared<Primitive>(kRealMakeTuple));

// Dict
GVAR_DEF(PrimitivePtr, kPrimDictLen, std::make_shared<Primitive>("dict_len"));
GVAR_DEF(PrimitivePtr, kPrimDictGetItem, std::make_shared<Primitive>("dict_getitem"));
GVAR_DEF(PrimitivePtr, kPrimDictSetItem, std::make_shared<Primitive>("dict_setitem"));
GVAR_DEF(PrimitivePtr, kPrimDictGetKeys, std::make_shared<Primitive>("dict_getkeys"));
GVAR_DEF(PrimitivePtr, kPrimDictGetValues, std::make_shared<Primitive>("dict_getvalues"));
GVAR_DEF(PrimitivePtr, kPrimDictItems, std::make_shared<Primitive>("dict_items"));

// Tuple
GVAR_DEF(PrimitivePtr, kPrimMakeTuple, std::make_shared<Primitive>(kMakeTuple));
GVAR_DEF(PrimitivePtr, kPrimTupleGetItem, std::make_shared<Primitive>(kTupleGetItem));
GVAR_DEF(PrimitivePtr, kPrimTupleSetItem, std::make_shared<Primitive>(kTupleSetItem));
GVAR_DEF(PrimitivePtr, kPrimTupleLessThan, std::make_shared<Primitive>(kTupleLt));
GVAR_DEF(PrimitivePtr, kPrimTupleLessEqual, std::make_shared<Primitive>(kTupleLe));
GVAR_DEF(PrimitivePtr, kPrimRealTupleGetItem, std::make_shared<Primitive>(kRealTupleGetItem));
GVAR_DEF(PrimitivePtr, kPrimTupleDiv, std::make_shared<Primitive>("tuple_div"));
GVAR_DEF(PrimitivePtr, kPrimTupleToArray, std::make_shared<Primitive>("tuple_to_array"));
GVAR_DEF(PrimitivePtr, kPrimListReduce, std::make_shared<Primitive>("list_reduce"));
GVAR_DEF(PrimitivePtr, kPrimTupleReversed, std::make_shared<Primitive>("tuple_reversed"));
GVAR_DEF(PrimitivePtr, kPrimTupleGreaterThan, std::make_shared<Primitive>("tuple_greater_than"));
GVAR_DEF(PrimitivePtr, kPrimTupleGreaterEqual, std::make_shared<Primitive>("tuple_greater_equal"));
GVAR_DEF(PrimitivePtr, kPrimListGreaterThan, std::make_shared<Primitive>("list_greater_than"));
GVAR_DEF(PrimitivePtr, kPrimTupleEqual, std::make_shared<Primitive>(kTupleEqual));

// List
GVAR_DEF(PrimitivePtr, kPrimListGetItem, std::make_shared<Primitive>(kListGetItem));
GVAR_DEF(PrimitivePtr, kPrimListSetItem, std::make_shared<Primitive>(kListSetItem));
GVAR_DEF(PrimitivePtr, kPrimListGreaterEqual, std::make_shared<Primitive>("list_greater_equal"));
GVAR_DEF(PrimitivePtr, kPrimMakeList, std::make_shared<Primitive>(kMakeListNew));
GVAR_DEF(PrimitivePtr, kPrimListLessThan, std::make_shared<Primitive>(kListLt));
GVAR_DEF(PrimitivePtr, kPrimListLessEqual, std::make_shared<Primitive>(kListLe));
GVAR_DEF(PrimitivePtr, kPrimListEqual, std::make_shared<Primitive>(kListEqual));
GVAR_DEF(PrimitivePtr, kPrimListDiff, std::make_shared<Primitive>(kListDiff));
GVAR_DEF(PrimitivePtr, kPrimListInplaceClear,
         std::make_shared<Primitive>(kListInplaceClear,
                                     mindspore::HashMap<std::string, ValuePtr>({{std::string(GRAPH_FLAG_SIDE_EFFECT_IO),
                                                                                 MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimListInplaceReverse,
         std::make_shared<Primitive>(kListInplaceReverse,
                                     mindspore::HashMap<std::string, ValuePtr>({{std::string(GRAPH_FLAG_SIDE_EFFECT_IO),
                                                                                 MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimListInplaceExtend,
         std::make_shared<Primitive>(kListInplaceExtend,
                                     mindspore::HashMap<std::string, ValuePtr>({{std::string(GRAPH_FLAG_SIDE_EFFECT_IO),
                                                                                 MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimListInplaceInsert,
         std::make_shared<Primitive>(kListInplaceInsert,
                                     mindspore::HashMap<std::string, ValuePtr>({{std::string(GRAPH_FLAG_SIDE_EFFECT_IO),
                                                                                 MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimListInplacePop,
         std::make_shared<Primitive>(kListInplacePop, mindspore::HashMap<std::string, ValuePtr>(
                                                        {{std::string(GRAPH_FLAG_SIDE_EFFECT_IO), MakeValue(true)}})));

// Sequence and Tensor
GVAR_DEF(PrimitivePtr, kPrimListToTensor, std::make_shared<Primitive>(kListToTensor));
GVAR_DEF(PrimitivePtr, kPrimTensorToTuple, std::make_shared<Primitive>(kTensorToTuple));
GVAR_DEF(PrimitivePtr, kPrimTensorToList, std::make_shared<Primitive>(kTensorToList));

// Sequence operations.
GVAR_DEF(PrimitivePtr, kPrimListAppend, std::make_shared<Primitive>(kListAppend));
GVAR_DEF(PrimitivePtr, kPrimListInplaceAppend, std::make_shared<Primitive>(kListInplaceAppend));
GVAR_DEF(PrimitivePtr, kPrimListInsert, std::make_shared<Primitive>(kListInsert));
GVAR_DEF(PrimitivePtr, kPrimListAppendAndInsertGrad, std::make_shared<Primitive>(kListAppendAndInsertGrad));
GVAR_DEF(PrimitivePtr, kPrimSequenceAdd, std::make_shared<Primitive>(kSequenceAdd));
GVAR_DEF(PrimitivePtr, kPrimSequenceCount, std::make_shared<Primitive>(kSequenceCount));
GVAR_DEF(PrimitivePtr, kPrimSequenceIndex, std::make_shared<Primitive>(kSequenceIndex));
GVAR_DEF(PrimitivePtr, kPrimSequenceMul, std::make_shared<Primitive>(kSequenceMul));
GVAR_DEF(PrimitivePtr, kPrimSequenceSlice, std::make_shared<Primitive>(kSequenceSlice));
GVAR_DEF(PrimitivePtr, kPrimSequenceSliceSetItem, std::make_shared<Primitive>(kSequenceSliceSetItem));
GVAR_DEF(PrimitivePtr, kPrimSequenceZerosLike, std::make_shared<Primitive>(kSequenceZerosLike));
GVAR_DEF(PrimitivePtr, kPrimSequenceAddOffset, std::make_shared<Primitive>(kSequenceAddOffset));
GVAR_DEF(PrimitivePtr, kPrimSequenceSliceGrad, std::make_shared<Primitive>(kSequenceSliceGrad));
GVAR_DEF(PrimitivePtr, kPrimSequenceMax, std::make_shared<Primitive>(kSequenceMax));
GVAR_DEF(PrimitivePtr, kPrimSequenceMin, std::make_shared<Primitive>(kSequenceMin));
GVAR_DEF(PrimitivePtr, kPrimInSequence, std::make_shared<Primitive>(kInSequence));
GVAR_DEF(PrimitivePtr, kPrimSequenceAddN, std::make_shared<Primitive>(kSequenceAddN));
GVAR_DEF(PrimitivePtr, kPrimSequenceConcat, std::make_shared<Primitive>(kSequenceConcat));
GVAR_DEF(PrimitivePtr, kPrimSequenceStack, std::make_shared<Primitive>(kSequenceStack));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_SEQUENCE_OPS_H_
