#include "trt_worker.hpp"
#include "trt_classifier.hpp"
#include "trt_detector.hpp"
#include "trt_logger.hpp"
#include "memory"

using namespace std;

namespace thread{

Worker::Worker(string onnxPath, logger::Level level, model::Params params) {
    m_logger = logger::create_logger(level);                                        //->创建日志等级实例

    if (params.task == model::task_type::CLASSIFICATION)                            //->图像分类任务
        m_classifier = model::classifier::make_classifier(onnxPath, level, params); //->创建图像分类任务实例
    else if (params.task == model::task_type::DETECTION)                            //->目标检测任务
        m_detector = model::detector::make_detector(onnxPath, level, params);       //->创建目标检测任务实例
}

void Worker::inference(string imagePath) {
    if (m_classifier != nullptr) {          //->图像分类任务
        m_classifier->init_model();         //->创建engine文件 or 加载engine文件
        m_classifier->load_image(imagePath);//->加载图像
        m_classifier->inference();          //->推理（前处理+推理+后处理）
    }

    if (m_detector != nullptr) {            //->目标检测任务
        m_detector->init_model();           //->创建目标检测engine文件 or 加载目标检测engine文件
        m_detector->load_image(imagePath);  //->加载图像
        m_detector->inference();            //->推理（前处理+推理+后处理）
    }
}

shared_ptr<Worker> create_worker(
    std::string onnxPath, logger::Level level, model::Params params) 
{
    return make_shared<Worker>(onnxPath, level, params);
}

}; //->namespace thread
