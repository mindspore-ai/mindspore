
#ifndef DATASET_INCLUDE_DETENSOR_H_
#define DATASET_INCLUDE_DETENSOR_H_
#include "include/ms_tensor.h"
#include "minddata/dataset/include/tensor.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace tensor {
class DETensor : public MSTensor {
 public:
    // brief Create a MSTensor pointer.
    //
    // param data_type DataTypeId of tensor to be created.
    // param shape Shape of tensor to be created.
    // return MSTensor pointer.
    static MSTensor *CreateTensor(TypeId data_type, const std::vector<int> &shape);

    static MSTensor *CreateTensor(const std::string &path);

    DETensor(TypeId data_type, const std::vector<int> &shape);

    explicit DETensor(std::shared_ptr<dataset::Tensor> tensor_ptr);

    ~DETensor() = default;

    MSTensor *ConvertToLiteTensor(); 

    std::shared_ptr<dataset::Tensor> tensor() const;

    TypeId data_type() const override;

    TypeId set_data_type(const TypeId data_type) override;

    std::vector<int> shape() const override;

    size_t set_shape(const std::vector<int> &shape) override;

    int DimensionSize(size_t index) const override;

    int ElementsNum() const override;

    std::size_t hash() const override;

    size_t Size() const override;

    void *MutableData() const override;

 protected:
    std::shared_ptr<dataset::Tensor> tensor_impl_;
};
}  // namespace tensor
}  // namespace mindspore
#endif  // DATASET_INCLUDE_DETENSOR_H_