#
# Copyright 2020-2022 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SHELL := /bin/bash
CPP = g++
CC = gcc
AR = ar
WGET = wget -r -q
MV = mv
CP = cp

MKDIR_P = mkdir -p
BUILD_DIR = build
LIB_DIR = $(OPENDR_HOME)/lib
DATA_DIR = ./data

CFLAGS  = -fPIC
INC = -I/usr/local/include/onnxruntime/core/session/ `pkg-config --cflags opencv4`
LD =  -L/usr/lib/ -L/usr/local/lib/ -lstdc++ -lm -lonnxruntime `pkg-config --libs opencv4` -lboost_filesystem
OPENDR_INC = -I$(OPENDR_HOME)/include
OPENDR_LD = -L$(OPENDR_HOME)/lib -lopendr

CMAKE = cmake
MAKE = make
INC_DIR = $(OPENDR_HOME)/include
CMAKE_OPENDR_TARGET = $(OPENDR_HOME)/build
LIBTORCH_DIR = /home/manos/nanodet/libtorchABI/libtorch

MY_DATA = $(OPENDR_HOME)/src/c_api/nanodet_libtorch
LIBTORCH_DIR = /home/manos/nanodet/libtorch
INC_TORCH = -I$(LIBTORCH_DIR)/include/torch/csrc/api/include -I$(LIBTORCH_DIR)/include
LD_TORCH = -L$(LIBTORCH_DIR)/lib -Lcaffe2.lib -Lc10.lib

all: download demos

demos: $(BUILD_DIR)/face_recognition_demo $(BUILD_DIR)/nanodet_demo

nano: $(BUILD_DIR)/nanodet_demo/gcc
face: $(BUILD_DIR)/face_recognition_demo

temp: $(BUILD_DIR)/nanodet_demo

download:
	@+if [ -a $(DATA_DIR) ] ; \
	then \
		echo Resources already downloaded. Run rm -rf $(DATA_DIR) to remove existing resources.; \
	else \
		echo Downloading resources for C API...; \
		$(RM) -r opendrdata.csd.auth.gr; \
		$(MKDIR_P) $(DATA_DIR); \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/face_recognition/test_data/*; \
		$(MV) opendrdata.csd.auth.gr/perception/face_recognition/test_data/images  $(DATA_DIR)/database; \
		$(WGET) ftp://opendrdata.csd.auth.gr/perception/face_recognition/optimized_model/*; \
		$(MV) opendrdata.csd.auth.gr/perception/face_recognition/optimized_model  $(DATA_DIR)/optimized_model; \
		$(RM) -r opendrdata.csd.auth.gr; \
	fi;

$(BUILD_DIR)/nanodet_demo/gcc:
	$(MKDIR_P) $(BUILD_DIR)
	@+echo "Building nanodet demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/nanodet_demo samples/nanodet/nanodet_demo.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD) $(LD_TORCH)
#$(CMAKE) -S ./samples/nanodet/ -B ./$(BUILD_DIR)/nanodet_demo -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR) -DOPENDR_LIB=$(LIB_DIR) -DOPENDR_INC=$(INC_DIR)
	$(CP) $(MY_DATA)/model/nanodet_m_0.5x.pth $(BUILD_DIR)/nanodet_demo
	$(CP) $(MY_DATA)/000000000037.jpg $(BUILD_DIR)/nanodet_demo
	#$(CMAKE) --build $(BUILD_DIR)/nanodet_demo

$(BUILD_DIR)/nanodet_demo:
	$(MKDIR_P) $(BUILD_DIR)
	@+echo "Building nanodet demo..."
	$(CMAKE) -S ./samples/nanodet/ -B ./$(BUILD_DIR)/nanodet_demo -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR) -DOPENDR_LIB=$(LIB_DIR) -DOPENDR_INC=$(INC_DIR)
	$(CP) $(MY_DATA)/model/nanodet_m_0.5x.pth $(BUILD_DIR)/nanodet_demo
	$(CP) $(MY_DATA)/000000000037.jpg $(BUILD_DIR)/nanodet_demo
	$(CMAKE) --build $(BUILD_DIR)/nanodet_demo

$(BUILD_DIR)/face_recognition_demo:
	$(MKDIR_P) $(BUILD_DIR)
	@+echo "Building face recognition demo..."
	$(CC)  $(CFLAGS) -o $(BUILD_DIR)/face_recognition_demo samples/face_recognition/face_recognition_demo.c $(INC) $(OPENDR_INC) $(OPENDR_LD) $(LD)

clean:
	@+echo "Cleaning C API demo binaries and temporary files..."
	@+$(RM) -r $(BUILD_DIR)
	@+$(RM) -rf $(DATA_DIR)
	@+echo "Done!"
