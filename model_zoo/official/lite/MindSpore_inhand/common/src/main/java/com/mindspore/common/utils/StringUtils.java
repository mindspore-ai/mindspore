package com.mindspore.common.utils;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class StringUtils {


    public static boolean checkChar(char ch) {
        return (ch + "").getBytes().length == 1;
    }

    public static boolean isChinese(String str) {
        String regEx1 = "[\\u4e00-\\u9fa5]+";
        String regEx2 = "[\\uFF00-\\uFFEF]+";
        String regEx3 = "[\\u2E80-\\u2EFF]+";
        String regEx4 = "[\\u3000-\\u303F]+";
        String regEx5 = "[\\u31C0-\\u31EF]+";
        Pattern p1 = Pattern.compile(regEx1);
        Pattern p2 = Pattern.compile(regEx2);
        Pattern p3 = Pattern.compile(regEx3);
        Pattern p4 = Pattern.compile(regEx4);
        Pattern p5 = Pattern.compile(regEx5);
        Matcher m1 = p1.matcher(str);
        Matcher m2 = p2.matcher(str);
        Matcher m3 = p3.matcher(str);
        Matcher m4 = p4.matcher(str);
        Matcher m5 = p5.matcher(str);
        return m1.find() || m2.find() || m3.find() || m4.find() || m5.find();
    }
}
