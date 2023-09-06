/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_SEQUENCE_OP_NAME_H_
#define MINDSPORE_CORE_BASE_SEQUENCE_OP_NAME_H_

namespace mindspore {
// Tuple
constexpr auto kRealMakeTupleOpName = "RealMakeTuple";
constexpr auto kMakeTupleOpName = "MakeTuple";
constexpr auto kTupleGetItemOpName = "TupleGetItem";
constexpr auto kTupleSetItemOpName = "tuple_setitem";
constexpr auto kTupleLtOpName = "tuple_lt";
constexpr auto kTupleLeOpName = "tuple_le";
constexpr auto kRealTupleGetItemOpName = "RealTupleGetItem";
constexpr auto kTupleGreaterThanOpName = "tuple_greater_than";
constexpr auto kTupleGreaterEqualOpName = "tuple_greater_equal";
constexpr auto kTupleEqualOpName = "tuple_equal";

// List
constexpr auto kListInplaceClearOpName = "ListInplaceClear";
constexpr auto kListInplaceReverseOpName = "ListInplaceReverse";
constexpr auto kListInplaceExtendOpName = "ListInplaceExtend";
constexpr auto kListInplaceInsertOpName = "ListInplaceInsert";
constexpr auto kListInplacePopOpName = "ListInplacePop";
constexpr auto kMakeListOpName = "MakeList";
constexpr auto kMakeListNewOpName = "make_list";
constexpr auto kListGetItemOpName = "list_getitem";
constexpr auto kListSetItemOpName = "list_setitem";
constexpr auto kListLtOpName = "list_lt";
constexpr auto kListLeOpName = "list_le";
constexpr auto kListGreaterThanOpName = "list_greater_than";
constexpr auto kListGreaterEqualOpName = "list_greater_equal";
constexpr auto kListEqualOpName = "list_equal";
constexpr auto kListDiffOpName = "ListDiff";

// Dict
constexpr auto kDictInplaceSetItemOpName = "DictInplaceSetItem";

// Sequence and Tensor
constexpr auto kTupleToTensorOpName = "TupleToTensor";
constexpr auto kTensorToTupleOpName = "TensorToTuple";
constexpr auto kListToTensorOpName = "ListToTensor";
constexpr auto kTensorToListOpName = "TensorToList";

// Sequence operation
constexpr auto kListAppendOpName = "ListAppend";
constexpr auto kListInsertOpName = "ListInsert";
constexpr auto kListInplaceAppendOpName = "ListInplaceAppend";
constexpr auto kListAppendAndInsertGradOpName = "ListAppendAndInsertGrad";
constexpr auto kSequenceAddOpName = "SequenceAdd";
constexpr auto kSequenceCountOpName = "SequenceCount";
constexpr auto kSequenceIndexOpName = "SequenceIndex";
constexpr auto kSequenceMulOpName = "SequenceMul";
constexpr auto kSequenceSliceOpName = "SequenceSlice";
constexpr auto kSequenceLenOpName = "sequence_len";
constexpr auto kSequenceZerosLikeOpName = "SequenceZerosLike";
constexpr auto kMakeRangeOpName = "make_range";
constexpr auto kSequenceAddOffsetOpName = "SequenceAddOffset";
constexpr auto kSequenceSliceGradOpName = "SequenceSliceGrad";
constexpr auto kSequenceSliceSetItemOpName = "SequenceSliceSetItem";
constexpr auto kSequenceMaxOpName = "SequenceMax";
constexpr auto kSequenceMinOpName = "SequenceMin";
constexpr auto kInSequenceOpName = "InSequence";
constexpr auto kSequenceAddNOpName = "SequenceAddN";
constexpr auto kSequenceConcatOpName = "SequenceConcat";
constexpr auto kSequenceStackOpName = "SequenceStack";
constexpr auto kSequenceUnstackOpName = "SequenceUnstack";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_SEQUENCE_OP_NAME_H_
