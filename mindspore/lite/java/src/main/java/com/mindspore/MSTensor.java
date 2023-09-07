/*
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

import static com.mindspore.config.MindsporeLite.POINTER_DEFAULT_VALUE;

import com.mindspore.config.MindsporeLite;
import com.mindspore.config.DataType;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.nio.FloatBuffer;
import java.lang.reflect.Array;
import java.util.HashMap;
import java.util.logging.Logger;
import javax.xml.crypto.Data;

/**
 * The MSTensor class defines a tensor in MindSpore.
 *
 * @since v1.0
 */
public class MSTensor {
    private static final Logger LOGGER = Logger.getLogger(MSTensor.class.toString());

    static {
        MindsporeLite.init();
    }

    private long tensorPtr;
    private Object buffer;

    /**
     * MSTensor construct function.
     */
    public MSTensor() {
        this.tensorPtr = POINTER_DEFAULT_VALUE;
        this.buffer = null;
    }

    /**
     * MSTensor construct function.
     *
     * @param tensorPtr tensor pointer.
     */
    public MSTensor(long tensorPtr) {
        this.tensorPtr = tensorPtr;
        this.buffer = null;
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
            LOGGER.severe("input params null.");
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
            LOGGER.severe("input params null.");
            return null;
        }
        int dType = ParseDataType(obj);
        if (dType == 0) {
            LOGGER.severe("input param dtype invalid.");
            return null;
        }
        int[] shape = ParseShape(obj);
        if (shape == null) {
            LOGGER.severe("input param shape null.");
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
     * Get output data of MSTensor, data type is the same as the type data is set.
     *
     * @return The byte array containing all MSTensor output data.
     */
    public Object getData() {
        Object ret = null;
        if (this.buffer != null) {
            return this.buffer;
        } else {
            int dataType = this.getDataType();
            switch(dataType) {
                case DataType.kNumberTypeFloat32:
                    ret = this.getFloatData(this.tensorPtr);
                    break;
                case DataType.kNumberTypeFloat16:
                    ret = this.getFloat16Data(this.tensorPtr);
                    break;
                case DataType.kNumberTypeInt32:
                    ret = this.getIntData(this.tensorPtr);
                    break;
                case DataType.kNumberTypeInt64:
                    ret = this.getLongData(this.tensorPtr);
                    break;
                default:
                    LOGGER.warning("Do not support data type: " + dataType + ", would return byte[] data");
                    ret = this.getByteData(this.tensorPtr);
            }
        }
        return ret;
    }

    /**
     * Get output data of MSTensor, the data type is byte.
     *
     * @return The byte array containing all MSTensor output data.
     */
    public byte[] getByteData() {
        if (this.buffer == null) {
            return this.getByteData(this.tensorPtr);
        }
        if (this.buffer instanceof byte[]) {
            return (byte[]) this.buffer;
        }
        byte[] ret = new byte[0];
        return ret;
    }

    /**
     * Get output data of MSTensor, the data type is float.
     *
     * @return The float array containing all MSTensor output data.
     */
    public float[] getFloatData() {
        if (this.buffer == null) {
            if (this.getDataType() == DataType.kNumberTypeFloat16) {
                return this.getFloat16Data(this.tensorPtr);
            }
            return this.getFloatData(this.tensorPtr);
        }
        if (this.buffer instanceof float[]) {
            return (float[]) this.buffer;
        }
        int dataType = this.getDataType();
        float[] floatArray = new float[0];
        if (this.buffer instanceof byte[]
            && (dataType == DataType.kNumberTypeFloat16 || dataType == DataType.kNumberTypeFloat32)) {
            ByteBuffer byteBuffer = ByteBuffer.wrap((byte[]) this.buffer);
            FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
            floatArray = new float[floatBuffer.remaining()];
            floatBuffer.get(floatArray);
        }
        return floatArray;
    }

    /**
     * Get output data of MSTensor, the data type is int.
     *
     * @return The int array containing all MSTensor output data.
     */
    public int[] getIntData() {
        if (this.buffer == null) {
            return this.getIntData(this.tensorPtr);
        }
        if (this.buffer instanceof int[]) {
            return (int[]) this.buffer;
        }
        int dataType = this.getDataType();
        int[] intArray = new int[0];
        if (this.buffer instanceof byte[]
            && (dataType == DataType.kNumberTypeInt32)) {
            byte[] byteArray = (byte[]) this.buffer;
            intArray = new int[byteArray.length];
            for (int i = 0; i < byteArray.length; i++) {
                intArray[i] = byteArray[i] & 0xff;
            }
        }
        return intArray;
    }

    /**
     * Get output data of MSTensor, the data type is long.
     *
     * @return The long array containing all MSTensor output data.
     */
    public long[] getLongData() {
        if (this.buffer == null) {
            return this.getLongData(this.tensorPtr);
        }
        if (this.buffer instanceof long[]) {
            return (long[]) this.buffer;
        }
        int dataType = this.getDataType();
        long[] longArray = new long[0];
        if (this.buffer instanceof byte[]
            && (dataType == DataType.kNumberTypeFloat16 || dataType == DataType.kNumberTypeFloat32)) {
            ByteBuffer byteBuffer = ByteBuffer.wrap((byte[]) this.buffer);
            LongBuffer longBuffer = byteBuffer.asLongBuffer();
            longArray = new long[longBuffer.remaining()];
            longBuffer.get(longArray);
        }
        return longArray;
    }

    /**
     * Set the shape of MSTensor.
     *
     * @param tensorShape of int[] type.
     * @return whether set shape success.
     */
    public boolean setShape(int[] tensorShape) {
        if (tensorShape == null) {
            LOGGER.severe("input param null.");
            return false;
        }
        return this.setShape(this.tensorPtr, tensorShape);
    }

    /**
     * Set the input data of MSTensor.
     *
     * @param data Input data of ByteBuffer type.
     * @return whether set data success.
     */
    public boolean setData(ByteBuffer data) {
        if (data == null) {
            LOGGER.severe("input param null.");
            return false;
        }
        return this.setByteBufferData(this.tensorPtr, data);
    }

    /**
     * Set the input data of MSTensor.
     *
     * @param data Input data of byte[] type.
     * @return whether set data success.
     */
    public boolean setData(byte[] data) {
        if (data == null) {
            LOGGER.severe("input param null.");
            return false;
        }
        if (data.length != this.size()) {
            return false;
        }
        this.buffer = data;
        return true;
    }

    /**
     * Set the input data of MSTensor.
     *
     * @param data Input data of float[] type.
     * @return whether set data success.
     */
    public boolean setData(float[] data) {
        if (data == null) {
            LOGGER.severe("input param null.");
            return false;
        }
        if (this.getDataType() != DataType.kNumberTypeFloat32 &&
            this.getDataType() != DataType.kNumberTypeFloat16) {
            LOGGER.severe("Data type is not consistent");
            return false;
        }
        if (data.length != this.elementsNum()) {
            return false;
        }
        this.buffer = data;
        return true;
    }

    /**
     * Set the input data of MSTensor.
     *
     * @param data Input data of int[] type.
     * @return whether set data success.
     */
    public boolean setData(int[] data) {
        if (data == null) {
            LOGGER.severe("input param null.");
            return false;
        }
        if (this.getDataType() != DataType.kNumberTypeInt32) {
            LOGGER.severe("Data type is not consistent");
            return false;
        }
        if (data.length != this.elementsNum()) {
            return false;
        }
        this.buffer = data;
        return true;
    }

    /**
     * Set the input data of MSTensor.
     *
     * @param data Input data of long[] type.
     * @return whether set data success.
     */
    public boolean setData(long[] data) {
        if (data == null) {
            LOGGER.severe("input param null.");
            return false;
        }
        if (this.getDataType() != DataType.kNumberTypeInt64) {
            LOGGER.severe("Data type is not consistent");
            return false;
        }
        if (data.length != this.elementsNum()) {
            return false;
        }
        this.buffer = data;
        return true;
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
        this.tensorPtr = POINTER_DEFAULT_VALUE;
        this.buffer = null;
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

    private native float[] getFloat16Data(long tensorPtr);

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