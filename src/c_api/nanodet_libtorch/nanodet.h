#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include "target.h"
#include "opendr_utils.h"

struct ObjectRect {
    int x;
    int y;
    int width;
    int height;
};

struct CenterPrior
{
    int x;
    int y;
    int stride;
};



class NanoDet
{
public:
    NanoDet(const char* modelPath, const int* inputSize, int regMax, std::vector<int> strides, torch::Device device);
    ~NanoDet();
    ObjectRect effectRoi();
    torch::jit::script::Module net;
//    std::vector<BoxInfo> detect(cv::Mat image, float scoreThreshold, float nmsThreshold, torch::Device);
//    std::vector<opendr_detection_target_t> detect(cv::Mat image, float scoreThreshold, float nmsThreshold, torch::Device);
    std::vector<opendr_detection_target_t> detect(opendr_image_t *image, float scoreThreshold, float nmsThreshold, torch::Device);
    int numClass = 80; // number of classes. 80 for COCO
    std::vector<std::string> labels{ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                     "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                     "hair drier", "toothbrush" };
    int inputSize[2]; // input height and width
    int regMax; // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides; // strides of the multi-level feature.

private:
    torch::Tensor mPreprocess(cv::Mat image);
    void mDecodeInfer(torch::Tensor clsPreds, torch::Tensor disPreds, std::vector<CenterPrior>& centerPriors, float threshold, std::vector<std::vector<opendr_detection_target_t>>& results);
//    BoxInfo mDisPred2Bbox(const float*& dflDet, int label, float score, int x, int y, int stride);
//    static void cNms(std::vector<BoxInfo>& result, float nmsThreshold);
    opendr_detection_target_t mDisPred2Bbox(const float*& dflDet, int label, float score, int x, int y, int stride);
    static void cNms(std::vector<opendr_detection_target_t>& result, float nmsThreshold);
    torch::Tensor meanValues;
    torch::Tensor normValues;
    ObjectRect effectRoiValue;
};
