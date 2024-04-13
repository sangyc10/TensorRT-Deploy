#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "trt_worker.hpp"
#include "utils.hpp"
#include <yaml-cpp/yaml.h>

using namespace std;

int main(int argc, char const *argv[])
{
    YAML::Node config = YAML::LoadFile("./config/config.yaml");                                          //->加载YAML配置文件

    string onnxPath       = config["onnxPath"].as<std::string>();                                        //->读取ONNX文件
    logger::Level level   = static_cast<logger::Level>(config["logging"]["level"].as<int32_t>());        //->设置日志等级
    model::Params params  = model::Params();
    const YAML::Node& img = config["model_params"]["image"];
    params.img       = {img[0].as<int>(), img[1].as<int>(), img[2].as<int>()};                           //->设置图像size
    params.num_cls   = config["model_params"]["num_cls"].as<int>();                                      //->设置图像分类类别数量
    params.task      = static_cast<model::task_type>(config["model_params"]["task"].as<int>());          //->设置任务类型
    params.dev       = static_cast<model::device>(config["model_params"]["device"].as<int>());           //->设置前/后处理设备
    params.prec      = static_cast<model::precision>(config["model_params"]["precision"].as<int>());     //->设置推理精度
    params.cal_list  = config["model_params"]["calibration_list"].as<string>();                          //->设置校准数据列表
    params.cal_table = config["model_params"]["calibration_table"].as<string>();                         //->校准参数缓存
    params.cal_batchsize = config["model_params"]["calibration_batchsize"].as<int>();                    //->设置校准batchsize

    auto worker   = thread::create_worker(onnxPath, level, params);                                      //->实例化推理模型

    if (config["images_path"]) {                                                                         //->读取配置文件中图像路径
        const YAML::Node& images_path = config["images_path"];
        if (images_path.IsSequence()) {
            for (size_t i = 0; i < images_path.size(); ++i) {
                string image_path = images_path[i].as<string>();
                worker->inference(image_path);                                                           //->推理
            }
        } else {
            LOGE("ERROR: 'images_path' key is not a sequence in the YAML file");
            return -1;
        }
    }
    else{
        LOGE("ERROR: 'images_path' key not found in YAML file");
        return -1;
    }

    return 0;
}

