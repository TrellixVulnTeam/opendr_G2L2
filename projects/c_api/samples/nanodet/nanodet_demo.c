#include <stdio.h>
#include "nanodet_c.h"

int main(int argc, char** argv)
{
    if (argc < 7)
    {
        fprintf(stderr, "usage: %s [model_path] [device] [images_path] [input_sizes] [reg_max] [strides]. \n model_path = path/to/your/libtorch/model.pth \n device = cuda or cpu \n images_path = \"xxx/xxx/*.jpg\" \n input_size = width height \n reg_max = reg_max size from your yaml config file \n strides = strides from your yaml config file \n", argv[0]);
        return -1;
    }

//    const int inputSize[2] = {atoi(argv[4]), atoi(argv[5])};
//    int regMax = atoi(argv[6]);
//
//    std::vector<int> strides;
//    for(int i=7; i<argc; i++)
//    {
//        strides.push_back(atoi(argv[i]));
//    }

    nanodet_model_t model;

    printf("start init model\n");
    load_nanodet_model(argv, argc, 0.4, 0.5, &model);
//    load_nanodet_model(argv[1], inputSize, regMax, strides, argv[2], 0.4, 0.5, &model);
    printf("success\n");

//    const char* images = argv[3];

//    objectRect_t effectRoi;

//    std::vector<cv::String> filenames;
//    cv::glob(images, filenames, false);

    opendr_image_t image;
//    for (std::string imgName : filenames)
//    {
//       load_image(imgName.c_str(), &image);
//            if (!image.data)
//            {
//                printf("Image not found!");
//                return 1;
//            }
//
//       opendr_detection_target_vector_t results;
//       results.data = infer_nanodet(&image, &model);
//
//       drawBboxes(&image, &model);
//       cv::waitKey(0);
//    }

    load_image(argv[3], &image);
    if (!image.data)
    {
        printf("Image not found!");
        return 1;
    }

//    opendr_detection_target_vector_t results;
//    results.data = infer_nanodet(&image, &model);
    infer_nanodet(&image, &model);

    drawBboxes(&image, &model);


    free_image(&image);

    return 0;

}
