/*
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

package com.mindspore;

import com.mindspore.lite.NativeLibrary;

import java.nio.ByteBuffer;

public class MSTensor {
    static {
        try {
            NativeLibrary.load();
        } catch (Exception e) {
            System.err.println("Failed to load MindSporLite native library.");
            e.printStackTrace();
            throw e;
        }
    }

    private long tensorPtr;

    /**
     * MSTensor construct function.
     */
    public MSTensor() {
        this.tensorPtr = 0;
    }

    /**
     * MSTensor construct function.
     *
     * @param tensorPtr tensor pointer.
     */
    public MSTensor(long tensorPtr) {
        this.tensorPtr = tensorPtr;
    }

    /**
     * MSTensor construct function.
     *
     * @param tensorName tensor name
     * @param buffer     tensor buffer
     */
    public static MSTensor createTensor(String tensorName, int dataType, int[] tensorShape, ByteBuffer buffer) {
        if (tensorName == null || tensorShape == null || buffer == null) {
            return null;
        }
        long tensorPtr = createTensorByNative(tensorName, dataType, tensorShape, buffer);
        return new MSTensor(tensorPtr);
    }

    /**
     * Get the shape of the MindSpore MSTensor.
     *
     * @return A array of int as the shape of the MindSpore MSTensor.
     */
    public int[] getShape() {
        return this.getShape(this.tensorPtr);
    }

    /**
     * DataType is defined in com.mindspore.DataType.
     *
     * @return The MindSpore data type of the MindSpore MSTensor class.
     */
    public int getDataType() {
        return this.getDataType(this.tensorPtr);
    }

    /**
     * Get output data of MSTensor, the data type is byte.
     *
     * @return The byte array containing all MSTensor output data.
     */
    public byte[] getByteData() {
        return this.getByteData(this.tensorPtr);
    }

    /**
     * Get output data of MSTensor, the data type is float.
     *
     * @return The float array containing all MSTensor output data.
     */
    public float[] getFloatData() {
        return this.getFloatData(this.tensorPtr);
    }

    /**
     * Get output data of MSTensor, the data type is int.
     *
     * @return The int array containing all MSTensor output data.
     */
    public int[] getIntData() {
        return this.getIntData(this.tensorPtr);
    }

    /**
     * Get output data of MSTensor, the data type is long.
     *
     * @return The long array containing all MSTensor output data.
     */
    public long[] getLongData() {
        return this.getLongData(this.tensorPtr);
    }

    /**
     * Set the input data of MSTensor.
     *
     * @param data Input data of byte[] type.
     * @return whether set data success.
     */
    public boolean setData(byte[] data) {
        if (data == null) {
            return false;
        }
        return this.setData(this.tensorPtr, data, data.length);
    }

    /**
     * Set the input data of MSTensor.
     *
     * @param data data Input data of ByteBuffer type
     * @return whether set data success.
     */
    public boolean setData(ByteBuffer data) {
        if (data == null) {
            return false;
        }
        return this.setByteBufferData(this.tensorPtr, data);
    }

    /**
     * Get the size of the data in MSTensor in bytes.
     *
     * @return The size of the data in MSTensor in bytes.
     */
    public long size() {
        return this.size(this.tensorPtr);
    }

    /**
     * Get the number of elements in MSTensor.
     *
     * @return The number of elements in MSTensor.
     */
    public int elementsNum() {
        return this.elementsNum(this.tensorPtr);
    }

    /**
     * Free all temporary memory in MindSpore MSTensor.
     */
    public void free() {
        this.free(this.tensorPtr);
        this.tensorPtr = 0;
    }

    /**
     * @return Get tensor name
     */
    public String tensorName() {
        return this.tensorName(this.tensorPtr);
    }

    /**
     * @return MSTensor pointer
     */
    public long getMSTensorPtr() {
        return tensorPtr;
    }

    private static native long createTensorByNative(String tensorName, int dataType, int[] tesorShape,
                                                    ByteBuffer buffer);

    private native int[] getShape(long tensorPtr);

    private native int getDataType(long tensorPtr);

    private native byte[] getByteData(long tensorPtr);

    private native long[] getLongData(long tensorPtr);

    private native int[] getIntData(long tensorPtr);

    private native float[] getFloatData(long tensorPtr);

    private native boolean setData(long tensorPtr, byte[] data, long dataLen);

    private native boolean setByteBufferData(long tensorPtr, ByteBuffer buffer);

    private native long size(long tensorPtr);

    private native int elementsNum(long tensorPtr);

    private native void free(long tensorPtr);

    private native String tensorName(long tensorPtr);
}