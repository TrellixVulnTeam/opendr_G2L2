# OpenDR Toolkit Reference Manual

*Release 1.1*
<div align="center">
  <img src="images/opendr_logo.png" />
</div>

Copyright &copy; 2020-2022 OpenDR Project.
OpenDR is funded from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449.

Permission to use, copy and distribute this documentation for any purpose and without fee is hereby granted in perpetuity, provided that no modifications are made to this documentation.

The copyright holder makes no warranty or condition, either expressed or implied, including but not limited to any implied warranties of merchantability and fitness for a particular purpose, regarding this manual and the associated software.
This manual is provided on an `as-is` basis.
Neither the copyright holder nor any applicable licensor will be liable for any incidental or consequential damages.

## Table of Contents

- Inference and Training API
    - `engine` Module
        - [engine.data Module](engine-data.md)
        - [engine.datasets Module](engine-datasets.md)
        - [engine.target Module](engine-target.md)
    - `perception` Module
        - face recognition:
            - [face_recognition_learner Module](face-recognition.md)
        - facial expression recognition:
            - [landmark_based_facial_expression_recognition](landmark-based-facial-expression-recognition.md)
        - pose estimation:
            - [lightweight_open_pose Module](lightweight-open-pose.md)
        - activity recognition:
            - [activity_recognition Module](activity-recognition.md)
        - action recognition:
            - [skeleton_based_action_recognition](skeleton-based-action-recognition.md)
        - speech recognition:
            - [matchboxnet Module](matchboxnet.md)
            - [edgespeechnets Module](edgespeechnets.md)
            - [quadraticselfonn Module](quadratic-selfonn.md)
        - object detection 2d:
            - [detr Module](detr.md)
            - [gem Module](gem.md)
            - [retinaface Module](face-detection-2d-retinaface.md)
            - [centernet Module](object-detection-2d-centernet.md)
            - [ssd Module](object-detection-2d-ssd.md)
            - [yolov3 Module](object-detection-2d-yolov3.md)
            - [seq2seq-nms Module](object-detection-2d-nms-seq2seq_nms.md)
        - object detection 3d:
            - [voxel Module](voxel-object-detection-3d.md)
        - object tracking 2d:
            - [fair_mot Module](object-tracking-2d-fair-mot.md)
            - [deep_sort Module](object-tracking-2d-deep-sort.md)
        - object tracking 3d:
            - [ab3dmot Module](object-tracking-3d-ab3dmot.md)
        - multimodal human centric:
            - [rgbd_hand_gesture_learner Module](rgbd-hand-gesture-learner.md)
            - [audiovisual_emotion_recognition_learner Module](audiovisual-emotion-recognition-learner.md)
        - compressive learning:
            - [multilinear_compressive_learning Module](multilinear-compressive-learning.md)
        - semantic segmentation:
            - [semantic_segmentation Module](semantic-segmentation.md)
        - panoptic segmentation:
            - [efficient_ps Module](efficient-ps.md)
        - heart anomaly detection:
            - [gated_recurrent_unit Module](gated-recurrent-unit-learner.md)
            - [attention_neural_bag_of_feature_learner Module](attention-neural-bag-of-feature-learner.md)
        - fall detection:
            - [fall_detection Module](fall-detection.md)

    - `control` Module
        - [mobile_manipulation Module](mobile-manipulation.md)
        - [single_demo_grasp Module](single-demonstration-grasping.md)

    - `simulation` Module
        - [human_model_generation Module](human_model_generation.md)
    - `data_generation` Module
        - [synthetic_facial_image_generation Module](synthetic_facial_image_generator.md)
        - [human_model_generation Module](human-model-generation.md)
    - `utils` Module
        - [Hyperparameter Tuning Module](hyperparameter_tuner.md)
    - `Stand-alone Utility Frameworks`
        - [Engine Agnostic Gym Environment with Reactive extension (EAGERx)](eagerx.md)
- [ROSBridge Package](rosbridge.md)
- [C Inference API](c-api.md)
    - [data.h](c-data-h.md)
    - [target.h](c-target-h.md)
    - [opendr_utils.h](c-opendr-utils-h.md)
    - [face_recognition.h](c-face-recognition-h.md)
- Projects
    - `C API` Module
        - [face recognition Demo](/projects/c_api)
    - `control` Module
        - [mobile_manipulation Demo](/projects/control/mobile_manipulation)
        - [single_demo_grasp Demo](/projects/control/single_demo_grasp)
    - `opendr workspace` Module
        - [opendr_ws](/projects/opendr_ws)
    - `perception` Module
        - activity recognition:
            - [activity_recognition Demo](/projects/perception/activity_recognition/demos/online_recognition)
        - face recognition:
            - [face_recognition_Demo](/projects/perception/face_recognition)
        - facial expression recognition:
            - [landmark_based_facial_expression_recognition Demo](/projects/perception/facial_expression_recognition/landmark_based_facial_expression_recognition)
        - heart anomaly detection:
            - [heart anomaly detection Demo](/projects/perception/heart_anomaly_detection)
        - pose estimation:
            - [lightweight_open_pose Demo](/projects/perception/lightweight_open_pose)
        - multimodal human centric:
            - [rgbd_hand_gesture_learner Demo](/projects/perception/multimodal_human_centric/rgbd_hand_gesture_recognition)
            - [audiovisual_emotion_recognition Demo](/projects/perception/multimodal_human_centric/audiovisual_emotion_recognition)
        - object detection 2d:
            - [detr Demo](/projects/perception/object_detection_2d/detr)
            - [gem Demo](/projects/perception/object_detection_2d/gem)
            - [retinaface Demo](/projects/perception/object_detection_2d/retinaface)
            - [centernet Demo](/projects/perception/object_detection_2d/centernet)
            - [ssd Demo](/projects/perception/object_detection_2d/ssd)
            - [yolov3 Demo](/projects/perception/object_detection_2d/yolov3)
            - [seq2seq-nms Demo](/projects/perception/object_detection_2d/nms/seq2seq-nms)
        - object detection 3d:
            - [voxel Demo](/projects/perception/object_detection_3d/demos/voxel_object_detection_3d)
        - object tracking 2d:
            - [fair_mot Demo](/projects/perception/object_tracking_2d/demos/fair_mot_deep_sort)
        - panoptic segmentation:
            - [efficient_ps Demo](/projects/perception/panoptic_segmentation/efficient_ps)
        - semantic segmentation:
            - [bisnet Demo](/projects/perception/semantic_segmentation/bisenet)
        - action recognition:
            - [skeleton_based_action_recognition Demo](/projects/perception/skeleton_based_action_recognition)
        - fall detection:
            - [fall_detection Demo](/projects/perception/fall_detection.md)
        - [full_map_posterior_slam Module](/projects/perception/slam/full_map_posterior_gmapping)
    - `simulation` Module
        - [SMPL+D Human Models Dataset](/projects/simulation/SMPL%2BD_human_models)
        - [Human-Data-Generation-Framework](/projects/simulation/human_dataset_generation)
        - [Human Model Generation Demos](/projects/simulation/human_dataset_generation)
    - `utils` Module
        - [Hyperparameter Tuning Module](/projects/utils/hyperparameter_tuner)
