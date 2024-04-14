#include "NvInfer.h"
#include "trt_calibrator.hpp"
#include "utils.hpp"
#include "trt_logger.hpp"
#include "trt_preprocess.hpp"

#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace std;
using namespace nvinfer1;

namespace model{

Int8EntropyCalibrator::Int8EntropyCalibrator(      //->calibrator的构造函数
    const int&    batchSize,
    const string& calibrationDataPath,
    const string& calibrationTablePath,
    const int&    inputSize,
    const int&    inputH,
    const int&    inputW,
    const int&    task_type):

    m_batchSize(batchSize),                         //->校准批次大小
    m_inputH(inputH),                               //->输入高
    m_inputW(inputW),                               //->输入宽
    m_inputSize(inputSize),                         //->像素点数 C*H*W
    m_inputCount(batchSize * inputSize),            //->一个批次的校准数据总像素点数
    m_calibrationTablePath(calibrationTablePath),   //->校准缓存文件路径
    m_task_type(task_type)                          //->任务类型
{
    m_imageList = loadDataList(calibrationDataPath);                                      //->加载校准文件路径，保存在vector中
    m_imageList.resize(static_cast<int>(m_imageList.size() / m_batchSize) * m_batchSize); //->校准文件数量调整为批次的整数倍
    std::random_shuffle(m_imageList.begin(), m_imageList.end(),                           //->校准图像随机分布
                        [](int i){ return rand() % i; });
    CUDA_CHECK(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));                 //->分配内存空间
}

bool Int8EntropyCalibrator::getBatch(
    void* bindings[], const char* names[], int nbBindings) noexcept
{
    if (m_imageIndex + m_batchSize >= m_imageList.size() + 1)       //->越界判断
        return false;

    LOG("%3d/%3d (%3dx%3d): %s", m_imageIndex + 1, m_imageList.size(), m_inputH, m_inputW, m_imageList.at(m_imageIndex).c_str());

    cv::Mat input_image;
    if (m_task_type == 1) {
        for (int i = 0; i < m_batchSize; i++) {             //->处理batch中所有的图片
            input_image = cv::imread(m_imageList.at(m_imageIndex++));
            preprocess::preprocess_resize_gpu(              //->图片处理，与真正推理的前处理做相同的处理
                input_image, 
                m_deviceInput + i * m_inputSize,
                m_inputH, m_inputW, 
                preprocess::tactics::GPU_BILINEAR_CENTER);
        }
    }
    else if (m_task_type == 0) {
        float mean[]       = {0.406, 0.456, 0.485};
        float std[]        = {0.225, 0.224, 0.229};
        for (int i = 0; i < m_batchSize; i ++){
            input_image = cv::imread(m_imageList.at(m_imageIndex++));
            preprocess::preprocess_resize_gpu(
                input_image, 
                m_deviceInput + i * m_inputSize,
                m_inputH, m_inputW, 
                mean, std, preprocess::tactics::GPU_BILINEAR);
        }
    }

    bindings[0] = m_deviceInput;

    return true;
}
    
const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept
{
    void* output;
    m_calibrationCache.clear();

    ifstream input(m_calibrationTablePath, ios::binary);
    input >> noskipws;
    if (m_readCache && input.good())
        copy(istream_iterator<char>(input), istream_iterator<char>(), back_inserter(m_calibrationCache));

    length = m_calibrationCache.size();
    if (length){
        LOG("Using cached calibration table to build INT8 trt engine...");
        output = &m_calibrationCache[0];
    }else{
        LOG("Creating new calibration table to build INT8 trt engine...");
        output = nullptr;
    }
    return output;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    ofstream output(m_calibrationTablePath, ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}

} //->namespace model
