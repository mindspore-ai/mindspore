package com.mindspore.flclient;


import com.alibaba.fastjson.JSON;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

/**
 * UT Case can be read from UT case file, using this class to read UT cases form file.
 * @author       : zhangzhaoju
 * @since  : 2022/4/14
 */
public class FLFrameWithModelTestCaseParser {
    private List<FLFrameWithModelTestCase> testCases;
    private String caseFilePath;

    FLFrameWithModelTestCaseParser(String path) {
        caseFilePath = path;
    }

    public List<Object[]> Parser() throws IOException {
        File caseFile = new File(caseFilePath);
        String strCase = FileUtils.readFileToString(caseFile);
        testCases = JSON.parseArray(strCase, FLFrameWithModelTestCase.class);
        List<Object[]> casePairs = new ArrayList<>();
        for (FLFrameWithModelTestCase tcase : testCases) {
            Object[] objs = new Object[2];
            objs[0] = tcase.getCaseName();
            objs[1] = tcase;
            casePairs.add(objs);
        }
        return casePairs;
    }

    public void testParser() throws IOException {
        Parser();
    }
}
