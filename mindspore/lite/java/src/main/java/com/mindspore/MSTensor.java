/*
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

import com.mindspore.config.MindsporeLite;
import com.mindspore.config.DataType;
import java.nio.ByteBuffer;
import java.lang.reflect.Array;
import java.util.HashMap;

/**
 * The MSTensor class defines a tensor in MindSpore.
 *
 * @since v1.0
 */
public class MSTensor {
    static {
        MindsporeLite.init();
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
        if (tensorName == null || tensorShape == null || buffer == null || dataType < DataType.kNumberTypeBool ||
            dataType > DataType.kNumberTypeFloat64) {
            return null;
        }
        long tensorPtr = createTensorByNative(tensorName, dataType, tensorShape, buffer);
        return new MSTensor(tensorPtr);
    }

    /**
     * MSTensor construct function.
     *
     * @param tensorName tensor name
     * @param obj        java Array or a Scalar. Support dtype: float, double, int, long, boolean.
     */
    public static MSTensor createTensor(String tensorName, Object obj) {
        if (tensorName == null || obj == null) {
            return null;
        }
        int dType = ParseDataType(obj);
        if (dType == 0) {
            return null;
        }
        int[] shape = ParseShape(obj);
        if (shape == null) {
            return null;
        }
        long tensorPtr = createTensorByObject(tensorName, dType, shape, obj);
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
     * Set the shape of MSTensor.
     *
     * @param shape of int[] type.
     * @return whether set shape success.
     */
    public boolean setShape(int[] tensorShape) {
        if (tensorShape == null) {
            return false;
        }
        return this.setShape(this.tensorPtr, tensorShape);
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
        return this.setByteData(this.tensorPtr, data, data.length);
    }

    /**
     * Set the input data of MSTensor.
     *
     * @param data Input data of float[] type.
     * @return whether set data success.
     */
    public boolean setData(float[] data) {
        if (data == null) {
            return false;
        }
        return this.setFloatData(this.tensorPtr, data, data.length);
    }

    /**
     * Set the input data of MSTensor.
     *
     * @param data Input data of int[] type.
     * @return whether set data success.
     */
    public boolean setData(int[] data) {
        if (data == null) {
            return false;
        }
        return this.setIntData(this.tensorPtr, data, data.length);
    }

    /**
     * Set the input data of MSTensor.
     *
     * @param data Input data of long[] type.
     * @return whether set data success.
     */
    public boolean setData(long[] data) {
        if (data == null) {
            return false;
        }
        return this.setLongData(this.tensorPtr, data, data.length);
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

    private static int ParseDataType(Object obj) {
        HashMap<Class<?>, Integer> classToDType = new HashMap<Class<?>, Integer>() {{
            put(float.class, DataType.kNumberTypeFloat32);
            put(Float.class, DataType.kNumberTypeFloat32);
            put(double.class, DataType.kNumberTypeFloat64);
            put(Double.class, DataType.kNumberTypeFloat64);
            put(int.class, DataType.kNumberTypeInt32);
            put(Integer.class, DataType.kNumberTypeInt32);
            put(long.class, DataType.kNumberTypeInt64);
            put(Long.class, DataType.kNumberTypeInt64);
            put(boolean.class, DataType.kNumberTypeBool);
            put(Boolean.class, DataType.kNumberTypeBool);
        }};
        Class<?> c = obj.getClass();
        while (c.isArray()) {
            c = c.getComponentType();
        }
        Integer dType = classToDType.get(c);
        return dType == null ? 0 : dType;
    }

    private static int[] ParseShape(Object obj) {
        int i = 0;
        Class<?> c = obj.getClass();
        while (c.isArray()) {
            c = c.getComponentType();
            ++i;
        }
        int[] shape = new int[i];
        i = 0;
        c = obj.getClass();
        while (c.isArray()) {
            shape[i] = Array.getLength(obj);
            if (shape[i] <= 0) {
                return null;
            }
            obj = Array.get(obj, 0);
            c = c.getComponentType();
            ++i;
        }
        return shape;
    }

    private static native long createTensorByNative(String tensorName, int dataType, int[] tesorShape,
                                                    ByteBuffer buffer);

    private static native long createTensorByObject(String tensorName, int dataType, int[] tesorShape,
                                                    Object obj);

    private native int[] getShape(long tensorPtr);

    private native int getDataType(long tensorPtr);

    private native byte[] getByteData(long tensorPtr);

    private native long[] getLongData(long tensorPtr);

    private native int[] getIntData(long tensorPtr);

    private native float[] getFloatData(long tensorPtr);

    private native boolean setByteData(long tensorPtr, byte[] data, long dataLen);

    private native boolean setFloatData(long tensorPtr, float[] data, long dataLen);

    private native boolean setIntData(long tensorPtr, int[] data, long dataLen);

    private native boolean setLongData(long tensorPtr, long[] data, long dataLen);

    private native boolean setShape(long tensorPtr, int[] tensorShape);

    private native boolean setByteBufferData(long tensorPtr, ByteBuffer buffer);

    private native long size(long tensorPtr);

    private native int elementsNum(long tensorPtr);

    private native void free(long tensorPtr);

    private native String tensorName(long tensorPtr);
}