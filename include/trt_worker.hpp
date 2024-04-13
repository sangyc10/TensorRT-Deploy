#ifndef __WORKER_HPP__
#define __WORKER_HPP__

#include <memory>
#include <vector>
#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "trt_classifier.hpp"
#include "trt_detector.hpp"

namespace thread{

class Worker {
public:
    Worker(std::string onnxPath, logger::Level level, model::Params params);
    void inference(std::string imagePath);

public:
    std::shared_ptr<logger::Logger>          m_logger;              //->日志实例
    std::shared_ptr<model::Params>           m_params;              //->参数

    std::shared_ptr<model::classifier::Classifier>  m_classifier;   //->保存图像分类任务实例
    std::shared_ptr<model::detector::Detector>      m_detector;     //->保存目标检测任务实例
    // std::vector<float>                              m_scores;
    // std::vector<model::detector::bbox>              m_boxes;
};

std::shared_ptr<Worker> create_worker(
    std::string onnxPath, logger::Level level, model::Params params);

}; //->namespace thread

#endif 
