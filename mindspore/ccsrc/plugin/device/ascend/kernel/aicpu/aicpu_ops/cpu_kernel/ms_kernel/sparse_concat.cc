/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#include "sparse_concat.h"
#include <iostream>
#include <string>
#include <vector>
#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "status.h"
using namespace std;
namespace {
const char *kSparseConcat = "SparseConcat";
const uint32_t kOutputNum = 3;
const uint32_t kInputNum = 3;
}  // namespace
namespace aicpu {
class MySparseTensor {
 public:
  MySparseTensor() : dims_(0) {}
  ~MySparseTensor() = default;
  uint32_t CreateSparseTensor(Tensor *ix, Tensor *vals, std::vector<int64_t> shape, std::vector<int64_t> order) {
    int64_t dims = ix->GetTensorShape()->GetDimSize(1);
    ix_ = std::make_shared<EigenTensor>(ix, ix->GetData());
    vals_ = std::make_shared<EigenTensor>(vals, vals->GetData());
    shape_.assign(shape.begin(), shape.end());
    order_.assign(order.begin(), order.end());
    dims_ = dims;
    return KERNEL_STATUS_OK;
  }
  class DimComparator {
   public:
    DimComparator(const TTypes<int64_t>::Matrix &ix, const std::vector<int64_t> &order,
                  const std::vector<int64_t> &shape)
        : ix_(ix), order_(order), dims_(shape.size()) {}

    inline bool operator()(const int64_t i, const int64_t j) const {
      for (int di = 0; di < dims_; ++di) {
        const int64_t d = order_[di];
        if (ix_(i, d) < ix_(j, d)) {
          return true;
        }
        if (ix_(i, d) > ix_(j, d)) {
          return false;
        }
      }
      return false;
    }

    // Compares two indices taken from corresponding index matrices, using the
    // standard, row-major (or lexicographic) order.  Useful for cases that need
    // to distinguish between all three orderings (<, ==, >).
    inline static int cmp(const TTypes<int64_t>::ConstMatrix &a_idx, const TTypes<int64_t>::ConstMatrix &b_idx,
                          const int64_t a_row, const int64_t b_row, const int dims) {
      for (int d = 0; d < dims; ++d) {
        const int64_t a = a_idx(a_row, d);
        const int64_t b = b_idx(b_row, d);
        if (a < b) {
          return -1;
        } else if (a > b) {
          return 1;
        }
      }
      return 0;
    }

   protected:
    const TTypes<int64_t>::Matrix ix_;
    const std::vector<int64_t> order_;
    const int dims_;
  };
  template <int ORDER_DIM>
  class FixedDimComparator : DimComparator {
   public:
    FixedDimComparator(const TTypes<int64_t>::Matrix &ix, const std::vector<int64_t> &order,
                       const std::vector<int64_t> &shape)
        : DimComparator(ix, order, shape) {}
    inline bool operator()(const int64_t i, const int64_t j) const {
      bool value = false;
      for (int di = 0; di < ORDER_DIM; ++di) {
        const int64_t d = order_[di];
        if (ix_(i, d) < ix_(j, d)) {
          value = true;
          break;
        }
        if (ix_(i, d) > ix_(j, d)) break;
      }
      return value;
    }
  };
  template <typename T>
  uint32_t Reorder(const std::vector<int64_t> &order) {
    KERNEL_CHECK_FALSE(order.size() == (std::size_t)dims_, KERNEL_STATUS_PARAM_INVALID,
                       "Order length must be SparseTensor rank");
    auto ix_t = ix_->matrix<int64_t>();
    auto vals_t = vals_->vec<T>();

    std::vector<int64_t> reorder(ix_->GetTensor()->GetTensorShape()->GetDimSize(0));
    std::iota(reorder.begin(), reorder.end(), 0);

    // Sort to get order of indices
    switch (order.size()) {
#define CASE_SORT(ORDER_SIZE)                                   \
  case ORDER_SIZE: {                                            \
    FixedDimComparator<ORDER_SIZE> sorter(ix_t, order, shape_); \
    std::sort(reorder.begin(), reorder.end(), sorter);          \
    break;                                                      \
  }
      CASE_SORT(0);
      CASE_SORT(1);
      CASE_SORT(2);
      CASE_SORT(3);
      CASE_SORT(4);
      CASE_SORT(5);
#undef CASE_SORT
      default: {
        DimComparator sorter(ix_t, order, shape_);
        std::sort(reorder.begin(), reorder.end(), sorter);
      }
    }

    // We have a forward reordering, but what we'll need is a
    // permutation (the inverse).  This can be calculated with O(1)
    // additional
    // and O(n) time (INVPERM) but we just do the simple thing here.
    std::vector<size_t> permutation(reorder.size());
    for (std::size_t n = 0; n < reorder.size(); ++n) {
      permutation[reorder[n]] = n;
    }

    // Update indices & values by converting the permutations to
    // a product of transpositions.  Iterate over the cycles in the
    // permutation, and convert each of those into a product of
    // transpositions (swaps):
    //   https://en.wikipedia.org/wiki/Cyclic_permutation
    // This is N swaps, 2*N comparisons.
    for (std::size_t n = 0; n + 1 < permutation.size(); ++n) {
      while (n != permutation[n]) {
        std::size_t r = permutation[n];
        std::swap_ranges(&(ix_t(n, 0)), &(ix_t(n + 1, 0)), &(ix_t(r, 0)));
        std::swap(vals_t(n), vals_t(r));
        std::swap(permutation[n], permutation[r]);
      }
    }

    order_.assign(order.begin(), order.end());
    return 0;
  }
  template <typename T>
  static MySparseTensor *Concat(const std::vector<MySparseTensor *> &tensors, Tensor *output_ix, Tensor *output_vals) {
    const int dims = tensors[0]->dims_;
    auto order_0 = tensors[0]->order_;
    const int primary_dim = order_0[0];
    std::vector<int64_t> final_order(order_0.begin(), order_0.end());
    std::vector<int64_t> final_shape(tensors[0]->shape_.begin(), tensors[0]->shape_.end());
    final_shape[primary_dim] = 0;  // We'll build this up as we go along.
    int num_entries = 0;

    bool fully_ordered = true;
    for (const MySparseTensor *st : tensors) {
      if (st->order_ != final_order) fully_ordered = false;
      const std::vector<int64_t> &st_shape = st->shape_;

      // Update dimension of final shape
      final_shape[primary_dim] = (final_shape[primary_dim] + st_shape[primary_dim]);

      num_entries += st->ix_->GetTensor()->GetTensorShape()->GetDimSize(0);  // Update number of entries
    }

    // If nonconsistent ordering among inputs, set final order to -1s.
    if (!fully_ordered) {
      final_order = std::vector<int64_t>(final_shape.size(), -1);
    }

    EigenTensor ixET(output_ix, output_ix->GetData());
    EigenTensor valsET(output_vals, output_vals->GetData());
    TTypes<int64_t>::Matrix ix_t = ixET.matrix<int64_t>();
    typename TTypes<T>::Vec vals_t = valsET.vec<T>();

    Eigen::DenseIndex offset = 0;
    int64_t shape_offset = 0;
    for (const MySparseTensor *st : tensors) {
      const int st_num_entries = st->ix_->GetTensor()->GetTensorShape()->GetDimSize(0);

      // Fill in indices & values.
      std::copy_n(&st->vals_->vec<T>()(0), st_num_entries, &vals_t(offset));

      const auto *st_ix = &st->ix_->matrix<int64_t>()(0, 0);
      auto *ix_out = &ix_t(offset, 0);
      for (int i = 0; i < st_num_entries * dims; ++i) {
        *ix_out++ = *st_ix++ + ((i % dims == primary_dim) ? shape_offset : 0);
      }

      offset += st_num_entries;
      shape_offset += st->shape_[primary_dim];
    }
    MySparseTensor *res = new MySparseTensor();
    res->CreateSparseTensor(output_ix, output_vals, final_shape, final_order);
    return res;
  }
  std::vector<int64_t> shape() { return shape_; };

 private:
  std::shared_ptr<EigenTensor> ix_;
  std::shared_ptr<EigenTensor> vals_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> order_;
  int32_t dims_;
};
template <typename T>
uint32_t DoCompute(CpuKernelContext &ctx) {
  int64_t concat_dim_attr_ = ctx.GetAttr("concat_dim") != NULL ? ctx.GetAttr("concat_dim")->GetInt() : 0;
  int64_t N = ctx.GetAttr("N") != NULL ? ctx.GetAttr("N")->GetInt() : 1;

  vector<Tensor *> inds;
  vector<Tensor *> vals;
  vector<Tensor *> shapes;

  vector<typename TTypes<int64_t>::Matrix> inds_t;
  vector<typename TTypes<T>::Vec> vals_t;
  vector<typename TTypes<int64_t>::Vec> shapes_t;
  for (int i = 0; i < N; i++) {
    Tensor *indice = ctx.Input(i);
    Tensor *value = ctx.Input(i + N);
    Tensor *shape = ctx.Input(i + N * 2);

    auto indice_shape = indice->GetTensorShape();
    const int indices_dim = 2;
    KERNEL_CHECK_FALSE(indice_shape->GetDims() == indices_dim, KERNEL_STATUS_PARAM_INVALID,
                       "Input indices should be a matrix but received shape %d at position %d", indice_shape->GetDims(),
                       i);

    auto value_shape = value->GetTensorShape();
    KERNEL_CHECK_FALSE(value_shape->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID,
                       "Input values should be a vector but received shape %d at position %d", value_shape->GetDims(),
                       i);

    auto shape_shape = shape->GetTensorShape();
    KERNEL_CHECK_FALSE(shape_shape->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID,
                       "Input shapes should be a vector but received shape %d at position %d", shape_shape->GetDims(),
                       i);

    int64_t ind_dim0 = indice_shape->GetDimSize(0);
    int64_t ind_dim1 = indice_shape->GetDimSize(1);
    int64_t val_dim0 = value_shape->GetDimSize(0);
    int64_t shape_dim0 = shape_shape->GetDimSize(0);

    KERNEL_CHECK_FALSE(ind_dim0 == val_dim0, KERNEL_STATUS_PARAM_INVALID,
                       "indices dim_size_0 [%lld] != values dim_size_0 [%lld] at position %d", ind_dim0, val_dim0, i);
    KERNEL_CHECK_FALSE(ind_dim1 == shape_dim0, KERNEL_STATUS_PARAM_INVALID,
                       "indices dim_size_1 [%lld] != shapes dim_size_0 [%lld] at position %d", ind_dim1, shape_dim0, i);

    EigenTensor indiceET(indice, indice->GetData());
    EigenTensor valueET(value, value->GetData());
    EigenTensor shapeET(shape, shape->GetData());
    inds_t.push_back(indiceET.matrix<int64_t>());
    vals_t.push_back(valueET.vec<T>());
    shapes_t.push_back(shapeET.vec<int64_t>());

    inds.push_back(indice);
    vals.push_back(value);
    shapes.push_back(shape);
  }
  const typename TTypes<int64_t>::Vec input_shape = shapes_t[0];
  const int input_rank = input_shape.size();
  const int concat_dim = (concat_dim_attr_ < 0) ? input_rank + concat_dim_attr_ : concat_dim_attr_;
  KERNEL_CHECK_FALSE(concat_dim >= 0 && concat_dim < input_rank, KERNEL_STATUS_PARAM_INVALID,
                     "Concat dimension must be in range [%d,%d),got %d", -input_rank, input_rank, concat_dim_attr_);
  for (int i = 1; i < N; i++) {
    const typename TTypes<int64_t>::Vec current_shape = shapes_t[i];
    KERNEL_CHECK_FALSE(current_shape.size() == input_rank, KERNEL_STATUS_PARAM_INVALID,
                       "Ranks of all input tensors must match: expected %d,but "
                       "got %d at position %d",
                       input_rank, current_shape.size(), i);
    for (int j = 0; j < input_rank; j++) {
      if (j != concat_dim) {
        KERNEL_CHECK_FALSE(input_shape(j) == current_shape(j), KERNEL_STATUS_PARAM_INVALID,
                           "Input shapes must match: expected %d for dimension "
                           "%d but got %d at position %d",
                           input_shape(j), j, current_shape(j), i);
      }
    }
  }
  vector<int64_t> std_order(input_rank);
  iota(std_order.begin(), std_order.end(), 0);

  vector<int64_t> concat_order;
  concat_order.reserve(input_rank);
  concat_order.push_back(concat_dim);
  for (int j = 0; j < input_rank; ++j) {
    if (j != concat_dim) {
      concat_order.push_back(j);
    }
  }
  vector<MySparseTensor *> sp_inputs;
  for (int i = 0; i < N; ++i) {
    vector<int64_t> current_shape;
    for (int j = 0; j < input_rank; j++) current_shape.push_back(shapes_t[i](j));
    MySparseTensor *tensor = new MySparseTensor();
    tensor->CreateSparseTensor(inds[i], vals[i], current_shape, std_order);
    sp_inputs.push_back(std::move(tensor));
    sp_inputs[i]->Reorder<T>(concat_order);
  }
  Tensor *output_ix = ctx.Output(0);
  Tensor *output_vals = ctx.Output(1);

  MySparseTensor *concat = MySparseTensor::Concat<T>(sp_inputs, output_ix, output_vals);
  concat->Reorder<T>(std_order);

  Tensor *output_shape_out = ctx.Output(2);
  EigenTensor output_shapeET(output_shape_out, output_shape_out->GetData());
  auto output_shape = output_shapeET.vec<int64_t>();
  auto concat_shape = concat->shape();
  for (std::size_t i = 0; i < concat_shape.size(); i++) {
    output_shape(i) = concat_shape[i];
  }
  return KERNEL_STATUS_OK;
}
uint32_t SparseConcatCpuKernel::Compute(CpuKernelContext &ctx) {
  int64_t N = ctx.GetAttr("N") != NULL ? ctx.GetAttr("N")->GetInt() : 1;
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, N * kInputNum, kOutputNum),
                      "SparseConcat check input and output number failed.");
  auto data_type = ctx.Input(N)->GetDataType();
  switch (data_type) {
    case DT_INT8:
      return DoCompute<int8_t>(ctx);
    case DT_UINT8:
      return DoCompute<uint8_t>(ctx);
    case DT_INT16:
      return DoCompute<int16_t>(ctx);
    case DT_UINT16:
      return DoCompute<uint16_t>(ctx);
    case DT_INT32:
      return DoCompute<int32_t>(ctx);
    case DT_UINT32:
      return DoCompute<uint32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t>(ctx);
    case DT_UINT64:
      return DoCompute<uint64_t>(ctx);
    case DT_FLOAT16:
      return DoCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return DoCompute<float>(ctx);
    case DT_BOOL:
      return DoCompute<bool>(ctx);
    case DT_DOUBLE:
      return DoCompute<double>(ctx);
    case DT_COMPLEX64:
      return DoCompute<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return DoCompute<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("SparseConcat kernel data type [%u] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
REGISTER_CPU_KERNEL(kSparseConcat, SparseConcatCpuKernel);
}  // namespace aicpu
