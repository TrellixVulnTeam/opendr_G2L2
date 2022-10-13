#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include "nanodet.h"


const int colorList[80][3] =
{
    //{255 ,255 ,255}, //bg
    {216 , 82 , 24},
    {236 ,176 , 31},
    {125 , 46 ,141},
    {118 ,171 , 47},
    { 76 ,189 ,237},
    {238 , 19 , 46},
    { 76 , 76 , 76},
    {153 ,153 ,153},
    {255 ,  0 ,  0},
    {255 ,127 ,  0},
    {190 ,190 ,  0},
    {  0 ,255 ,  0},
    {  0 ,  0 ,255},
    {170 ,  0 ,255},
    { 84 , 84 ,  0},
    { 84 ,170 ,  0},
    { 84 ,255 ,  0},
    {170 , 84 ,  0},
    {170 ,170 ,  0},
    {170 ,255 ,  0},
    {255 , 84 ,  0},
    {255 ,170 ,  0},
    {255 ,255 ,  0},
    {  0 , 84 ,127},
    {  0 ,170 ,127},
    {  0 ,255 ,127},
    { 84 ,  0 ,127},
    { 84 , 84 ,127},
    { 84 ,170 ,127},
    { 84 ,255 ,127},
    {170 ,  0 ,127},
    {170 , 84 ,127},
    {170 ,170 ,127},
    {170 ,255 ,127},
    {255 ,  0 ,127},
    {255 , 84 ,127},
    {255 ,170 ,127},
    {255 ,255 ,127},
    {  0 , 84 ,255},
    {  0 ,170 ,255},
    {  0 ,255 ,255},
    { 84 ,  0 ,255},
    { 84 , 84 ,255},
    { 84 ,170 ,255},
    { 84 ,255 ,255},
    {170 ,  0 ,255},
    {170 , 84 ,255},
    {170 ,170 ,255},
    {170 ,255 ,255},
    {255 ,  0 ,255},
    {255 , 84 ,255},
    {255 ,170 ,255},
    { 42 ,  0 ,  0},
    { 84 ,  0 ,  0},
    {127 ,  0 ,  0},
    {170 ,  0 ,  0},
    {212 ,  0 ,  0},
    {255 ,  0 ,  0},
    {  0 , 42 ,  0},
    {  0 , 84 ,  0},
    {  0 ,127 ,  0},
    {  0 ,170 ,  0},
    {  0 ,212 ,  0},
    {  0 ,255 ,  0},
    {  0 ,  0 , 42},
    {  0 ,  0 , 84},
    {  0 ,  0 ,127},
    {  0 ,  0 ,170},
    {  0 ,  0 ,212},
    {  0 ,  0 ,255},
    {  0 ,  0 ,  0},
    { 36 , 36 , 36},
    { 72 , 72 , 72},
    {109 ,109 ,109},
    {145 ,145 ,145},
    {182 ,182 ,182},
    {218 ,218 ,218},
    {  0 ,113 ,188},
    { 80 ,182 ,188},
    {127 ,127 ,  0},
};

//void drawBboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, ObjectRect effectRoi, std::vector<std::string> classNames)
void drawBboxes(opendr_image_t *opendr_image, const std::vector<opendr_detection_target_t>& bboxes, ObjectRect effectRoi, std::vector<std::string> classNames)
{
    cv::Mat *opencv_image = static_cast<cv::Mat *>(opendr_image->data);
    if (!opencv_image) {
      std::cerr << "Cannot load image for inference." << std::endl;
      return;
    }

    cv::Mat image = (*opencv_image).clone();
    int srcW = image.cols;
    int srcH = image.rows;
    int dstW = effectRoi.width;
    int dstH = effectRoi.height;
    float widthRatio = (float)srcW / (float)dstW;
    float heightRatio = (float)srcH / (float)dstH;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const opendr_detection_target_t& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(colorList[bbox.name][0], colorList[bbox.name][1], colorList[bbox.name][2]);

        cv::rectangle(image, cv::Rect(cv::Point((bbox.left - effectRoi.x) * widthRatio, (bbox.top - effectRoi.y) * heightRatio),
                                      cv::Point(((bbox.left + bbox.width) - effectRoi.x) * widthRatio, ((bbox.top + bbox.height) - effectRoi.y) * heightRatio)), color);

        char text[256];
        float score = bbox.score > 1 ? 1 : bbox.score;
        sprintf(text, "%s %.1f%%", classNames[bbox.name].c_str(), score * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (bbox.left - effectRoi.x) * widthRatio;
        int y = (bbox.top - effectRoi.y) * heightRatio - labelSize.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + labelSize.width > image.cols)
            x = image.cols - labelSize.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
            color, -1);

        cv::putText(image, text, cv::Point(x, y + labelSize.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    cv::imshow("image", image);
}

int main(int argc, char** argv)
{
    if (argc < 7)
    {
        fprintf(stderr, "usage: %s [model_path] [device] [images_path] [input_sizes] [reg_max] [strides]. \n model_path = path/to/your/libtorch/model.pth \n device = cuda or cpu \n images_path = \"xxx/xxx/*.jpg\" \n input_size = width height \n reg_max = reg_max size from your yaml config file \n strides = strides from your yaml config file \n", argv[0]);
        return -1;
    }

    torch::DeviceType deviceType;
    if (std::string(argv[2]) == "cuda")
    {
        fprintf(stderr, "to cuda\n");
        deviceType = torch::kCUDA;
    }
    else
    {
        fprintf(stderr, "to cpu\n");
        deviceType = torch::kCPU;
    }
    torch::Device device(deviceType, 0);

    const int inputSize[2] = {atoi(argv[4]), atoi(argv[5])};
    int regMax = atoi(argv[6]);

    std::vector<int> strides;
    for(int i=7; i<argc; i++)
    {
        strides.push_back(atoi(argv[i]));
    }

    std::cout<<"start init model"<<std::endl;
    NanoDet detector = NanoDet(argv[1], inputSize, regMax, strides, device);
    std::cout<<"success"<<std::endl;

    const char* images = argv[3];

    ObjectRect effectRoi;

    std::vector<cv::String> filenames;
    cv::glob(images, filenames, false);
    int height = detector.inputSize[0];
    int width = detector.inputSize[1];

    opendr_image_t image;
    for (std::string imgName : filenames)
    {
        load_image(imgName.c_str(), &image);
        if (!image.data)
        {
            printf("Image not found!");
            return 1;
        }

        std::vector<opendr_detection_target_t> results = detector.detect(&image, 0.4, 0.5, device);

        effectRoi = detector.effectRoi();

        drawBboxes(&image, results, effectRoi, detector.labels);
        cv::waitKey(0);
    }
    free_image(&image);

    return 0;

}
