DLManagedTensor* toDLPack(constTensor& src){
ATenDLMTensor * atDLMTensor(newATenDLMTensor);

atDLMTensor->handle = src;

atDLMTensor->tensor.manager_ctx = atDLMTensor;

atDLMTensor->tensor.deleter = &deleter;

atDLMTensor->tensor.dl_tensor.data = src.data_ptr();

int64_tdevice_id = 0;

if(src.type().is_cuda()) {
device_id = src.get_device();

}

atDLMTensor->tensor.dl_tensor.ctx = getDLContext(src.type(), device_id);

atDLMTensor->tensor.dl_tensor.ndim = src.dim(
);
atDLMTensor->tensor.dl_tensor.dtype = getDLDataType(src.type());

atDLMTensor->tensor.dl_tensor.shape = const_cast< int64_t*>(src.sizes().data());

atDLMTensor->tensor.dl_tensor.strides = const_cast< int64_t*>(src.strides().data());

atDLMTensor->tensor.dl_tensor.byte_offset = 0;

return&(atDLMTensor->tensor);

}
