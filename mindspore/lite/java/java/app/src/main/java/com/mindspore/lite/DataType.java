package com.mindspore.lite;

public class DataType {
    public static final int kNumberTypeBool = 30;
    public static final int kNumberTypeInt = 31;
    public static final int kNumberTypeInt8 = 32;
    public static final int kNumberTypeInt16 = 33;
    public static final int kNumberTypeInt32 = 34;
    public static final int kNumberTypeInt64 = 35;
    public static final int kNumberTypeUInt = 36;
    public static final int kNumberTypeUInt8 = 37;
    public static final int kNumberTypeUInt16 = 38;
    public static final int kNumberTypeUint32 = 39;
    public static final int kNumberTypeUInt64 = 40;
    public static final int kNumberTypeFloat = 41;
    public static final int kNumberTypeFloat16 = 42;
    public static final int kNumberTypeFloat32 = 43;
    public static final int kNumberTypeFloat64 = 44;

    public static native int elementSize(int elementType);
}
