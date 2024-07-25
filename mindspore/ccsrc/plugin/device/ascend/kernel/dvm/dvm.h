/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef _DVM_H_
#define _DVM_H_

#include <cstdint>
#include <vector>

namespace dvm {
enum DType {
  kInt8 = 0,
  kFloat16,
  kBFloat16,
  kFloat32,
  kInt32,
  kTypeEnd,
};

enum UnaryOpType {
  kSqrt = 0,
  kAbs,
  kLog,
  kExp,
  kReciprocal,
  kIsFinite,
  kLogicalNot,
  kUnaryOpEnd,
};

enum BinaryOpType {
  kEqual = 0,
  kNotEqual,
  kGreater,
  kGreaterEqual,
  kLess,
  kLessEqual,
  kAdd,
  kSub,
  kMul,
  kDiv,
  kPow,
  kMaximum,
  kMinimum,
  kLogicalAnd,
  kLogicalOr,
  kBinaryOpEnd,
};

enum ReduceOpType {
  kSum = 0,
  kReduceOpEnd,
};

enum KernelType {
  kStaticShape = 0,
  kDynShape,
  kStaticParallel,
  kStaticMix,
  kStaticStages,
  kEager,
  kKernelTypelEnd,
};

class NDObject;
class VKernel;
class MsProfHelper;

struct ShapeRef {
  ShapeRef() {}
  explicit ShapeRef(const std::vector<int64_t> &other) : data(other.data()), size(other.size()) {}
  ShapeRef &operator=(const std::vector<int64_t> &other) {
    data = other.data();
    size = other.size();
    return *this;
  }
  const int64_t *data;
  size_t size;
};

struct RelocTable {
  NDObject **inputs;
  size_t inputs_size;
  NDObject **outputs;
  size_t outputs_size;
};

class Kernel {
 public:
  Kernel();
  ~Kernel();

  void Reset(KernelType type);
  void Reserve(size_t size);
  int ParallelNext();

  NDObject *Load(void *addr, ShapeRef *shape, DType type);
  NDObject *SliceLoad(void *addr, ShapeRef *shape, ShapeRef *start, ShapeRef *size, DType type);
  NDObject *StridedSliceLoad(void *addr, ShapeRef *shape, ShapeRef *start, ShapeRef *end, ShapeRef *step, DType type);
  NDObject *Store(void *addr, NDObject *input);
  NDObject *PadStore(void *addr, NDObject *input, ShapeRef *pad_shape);

  NDObject *Unary(int op_type, NDObject *input);
  NDObject *Binary(int op_type, NDObject *lhs, NDObject *rhs);
  template <typename T>
  NDObject *Binary(int op_type, T val, NDObject *rhs);
  template <typename T>
  NDObject *Binary(int op_type, NDObject *lhs, T val);

  NDObject *Reduce(int op_type, NDObject *input, ShapeRef *dims, bool keepdims);
  NDObject *Select(NDObject *cond, NDObject *lhs, NDObject *rhs);

  NDObject *Cast(NDObject *input, DType type);
  NDObject *Broadcast(NDObject *input, ShapeRef *shape);

  template <typename T>
  NDObject *Broadcast(T val, ShapeRef *shape, DType type, bool dummy_load);
  NDObject *Reshape(NDObject *input, ShapeRef *shape);
  NDObject *Copy(NDObject *input);

  NDObject *ElemAny(NDObject *input);

  NDObject *MatMul(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b);

  void StageSwitch(KernelType type);
  NDObject *StageLoad(NDObject *stage_store);
  NDObject *StageStore(NDObject *input);
  NDObject *StagePadStore(NDObject *input, ShapeRef *pad_shape);

  uint64_t CodeGen();
  int Launch(void *workspace, void *stream);
  int Launch(const RelocTable &reloc_table, void **inputs, void **outputs, void *workspace, void *stream);
  int Launch(NDObject **op, int size, void *stream);
  int MsProfLaunch(const char *op_name, const char *op_fullname, const RelocTable &reloc_table, void **inputs,
                   void **outputs, void *workspace, void *stream);

  ShapeRef *GetShape(NDObject *op) const;
  DType GetDType(NDObject *op) const;

  const char *Dump() const;
  const char *Das() const;

  VKernel *GetImpl() const { return kernel_; }

 private:
  VKernel *kernel_;
  MsProfHelper *msprof_helper_;
};

void SetDeterministic(bool enable);

}  // namespace dvm
#endif  // _DVM_H_
