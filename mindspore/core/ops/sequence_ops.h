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
#include "ops/sequence_op_name.h"
#include "utils/flags.h"

namespace mindspore {
namespace prim {
GVAR_DEF(PrimitivePtr, kPrimSequenceLen, std::make_shared<Primitive>("sequence_len"));

// Array
GVAR_DEF(PrimitivePtr, kPrimArrayLen, std::make_shared<Primitive>("array_len"));
GVAR_DEF(PrimitivePtr, kPrimArrayGetItem, std::make_shared<Primitive>("array_getitem"));
GVAR_DEF(PrimitivePtr, kPrimArraySetItem, std::make_shared<Primitive>("array_setitem"));

// Real tuple and list ops.
GVAR_DEF(PrimitivePtr, kPrimTupleToTensor, std::make_shared<Primitive>(kTupleToTensorOpName));
GVAR_DEF(PrimitivePtr, kPrimRealMakeTuple, std::make_shared<Primitive>(kRealMakeTupleOpName));

// Dict
GVAR_DEF(PrimitivePtr, kPrimDictLen, std::make_shared<Primitive>("dict_len"));
GVAR_DEF(PrimitivePtr, kPrimDictGetItem, std::make_shared<Primitive>("dict_getitem"));
GVAR_DEF(PrimitivePtr, kPrimDictSetItem, std::make_shared<Primitive>("dict_setitem"));
GVAR_DEF(PrimitivePtr, kPrimDictGetKeys, std::make_shared<Primitive>("dict_getkeys"));
GVAR_DEF(PrimitivePtr, kPrimDictGetValues, std::make_shared<Primitive>("dict_getvalues"));
GVAR_DEF(PrimitivePtr, kPrimDictItems, std::make_shared<Primitive>("dict_items"));
GVAR_DEF(PrimitivePtr, kPrimDictInplaceSetItem,
         std::make_shared<Primitive>(kDictInplaceSetItemOpName,
                                     mindspore::HashMap<std::string, ValuePtr>({{std::string(GRAPH_FLAG_SIDE_EFFECT_IO),
                                                                                 MakeValue(true)}})));

// Tuple
GVAR_DEF(PrimitivePtr, kPrimMakeTuple, std::make_shared<Primitive>(kMakeTupleOpName));
GVAR_DEF(PrimitivePtr, kPrimTupleGetItem, std::make_shared<Primitive>(kTupleGetItemOpName));
GVAR_DEF(PrimitivePtr, kPrimTupleSetItem, std::make_shared<Primitive>(kTupleSetItemOpName));
GVAR_DEF(PrimitivePtr, kPrimTupleLessThan, std::make_shared<Primitive>(kTupleLtOpName));
GVAR_DEF(PrimitivePtr, kPrimTupleLessEqual, std::make_shared<Primitive>(kTupleLeOpName));
GVAR_DEF(PrimitivePtr, kPrimRealTupleGetItem, std::make_shared<Primitive>(kRealTupleGetItemOpName));
GVAR_DEF(PrimitivePtr, kPrimTupleDiv, std::make_shared<Primitive>("tuple_div"));
GVAR_DEF(PrimitivePtr, kPrimTupleToArray, std::make_shared<Primitive>("tuple_to_array"));
GVAR_DEF(PrimitivePtr, kPrimListReduce, std::make_shared<Primitive>("list_reduce"));
GVAR_DEF(PrimitivePtr, kPrimTupleReversed, std::make_shared<Primitive>("tuple_reversed"));
GVAR_DEF(PrimitivePtr, kPrimTupleGreaterThan, std::make_shared<Primitive>("tuple_greater_than"));
GVAR_DEF(PrimitivePtr, kPrimTupleGreaterEqual, std::make_shared<Primitive>("tuple_greater_equal"));
GVAR_DEF(PrimitivePtr, kPrimListGreaterThan, std::make_shared<Primitive>("list_greater_than"));
GVAR_DEF(PrimitivePtr, kPrimTupleEqual, std::make_shared<Primitive>(kTupleEqualOpName));

// List
GVAR_DEF(PrimitivePtr, kPrimListGetItem, std::make_shared<Primitive>(kListGetItemOpName));
GVAR_DEF(PrimitivePtr, kPrimListSetItem, std::make_shared<Primitive>(kListSetItemOpName));
GVAR_DEF(PrimitivePtr, kPrimListGreaterEqual, std::make_shared<Primitive>("list_greater_equal"));
GVAR_DEF(PrimitivePtr, kPrimMakeList, std::make_shared<Primitive>(kMakeListNewOpName));
GVAR_DEF(PrimitivePtr, kPrimListLessThan, std::make_shared<Primitive>(kListLtOpName));
GVAR_DEF(PrimitivePtr, kPrimListLessEqual, std::make_shared<Primitive>(kListLeOpName));
GVAR_DEF(PrimitivePtr, kPrimListEqual, std::make_shared<Primitive>(kListEqualOpName));
GVAR_DEF(PrimitivePtr, kPrimListDiff, std::make_shared<Primitive>(kListDiffOpName));
GVAR_DEF(PrimitivePtr, kPrimListInplaceClear,
         std::make_shared<Primitive>(kListInplaceClearOpName,
                                     mindspore::HashMap<std::string, ValuePtr>({{std::string(GRAPH_FLAG_SIDE_EFFECT_IO),
                                                                                 MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimListInplaceReverse,
         std::make_shared<Primitive>(kListInplaceReverseOpName,
                                     mindspore::HashMap<std::string, ValuePtr>({{std::string(GRAPH_FLAG_SIDE_EFFECT_IO),
                                                                                 MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimListInplaceExtend,
         std::make_shared<Primitive>(kListInplaceExtendOpName,
                                     mindspore::HashMap<std::string, ValuePtr>({{std::string(GRAPH_FLAG_SIDE_EFFECT_IO),
                                                                                 MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimListInplaceInsert,
         std::make_shared<Primitive>(kListInplaceInsertOpName,
                                     mindspore::HashMap<std::string, ValuePtr>({{std::string(GRAPH_FLAG_SIDE_EFFECT_IO),
                                                                                 MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimListInplacePop,
         std::make_shared<Primitive>(kListInplacePopOpName,
                                     mindspore::HashMap<std::string, ValuePtr>({{std::string(GRAPH_FLAG_SIDE_EFFECT_IO),
                                                                                 MakeValue(true)}})));

// Sequence and Tensor
GVAR_DEF(PrimitivePtr, kPrimListToTensor, std::make_shared<Primitive>(kListToTensorOpName));
GVAR_DEF(PrimitivePtr, kPrimTensorToTuple, std::make_shared<Primitive>(kTensorToTupleOpName));
GVAR_DEF(PrimitivePtr, kPrimTensorToList, std::make_shared<Primitive>(kTensorToListOpName));

// Sequence operations.
GVAR_DEF(PrimitivePtr, kPrimListAppend, std::make_shared<Primitive>(kListAppendOpName));
GVAR_DEF(PrimitivePtr, kPrimListInplaceAppend, std::make_shared<Primitive>(kListInplaceAppendOpName));
GVAR_DEF(PrimitivePtr, kPrimListInsert, std::make_shared<Primitive>(kListInsertOpName));
GVAR_DEF(PrimitivePtr, kPrimListAppendAndInsertGrad, std::make_shared<Primitive>(kListAppendAndInsertGradOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceAdd, std::make_shared<Primitive>(kSequenceAddOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceCount, std::make_shared<Primitive>(kSequenceCountOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceIndex, std::make_shared<Primitive>(kSequenceIndexOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceMul, std::make_shared<Primitive>(kSequenceMulOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceSlice, std::make_shared<Primitive>(kSequenceSliceOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceSliceSetItem, std::make_shared<Primitive>(kSequenceSliceSetItemOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceZerosLike, std::make_shared<Primitive>(kSequenceZerosLikeOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceAddOffset, std::make_shared<Primitive>(kSequenceAddOffsetOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceSliceGrad, std::make_shared<Primitive>(kSequenceSliceGradOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceMax, std::make_shared<Primitive>(kSequenceMaxOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceMin, std::make_shared<Primitive>(kSequenceMinOpName));
GVAR_DEF(PrimitivePtr, kPrimInSequence, std::make_shared<Primitive>(kInSequenceOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceAddN, std::make_shared<Primitive>(kSequenceAddNOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceConcat, std::make_shared<Primitive>(kSequenceConcatOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceStack, std::make_shared<Primitive>(kSequenceStackOpName));
GVAR_DEF(PrimitivePtr, kPrimSequenceUnstack, std::make_shared<Primitive>(kSequenceUnstackOpName));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_SEQUENCE_OPS_H_
