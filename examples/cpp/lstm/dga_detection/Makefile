# This is a simple Makefile used to build the example source code.
# This example might requires some modifications in order to work correctly on
# your system.
# If you're not using the Armadillo wrapper, replace `armadillo` with linker commands
# for the BLAS and LAPACK libraries that you are using.

TARGET1 := lstm_dga_detection_train
SRC1 := lstm_dga_detection_train.cpp
TARGET2 := lstm_dga_detection_predict
SRC2 := lstm_dga_detection_predict.cpp
LIBS_NAME := armadillo

CXX := g++
CXXFLAGS += -std=c++17 -Wall -Wextra -O3 -DNDEBUG -fopenmp
# Use these CXXFLAGS instead if you want to compile with debugging symbols and
# without optimizations.
#CXXFLAGS += -std=c++17 -Wall -Wextra -g -O0 -fopenmp
LDFLAGS  += -fopenmp
# Add header directories for any includes that aren't on the
# default compiler search path.
INCLFLAGS := -I .
# If you have mlpack or ensmallen installed somewhere nonstandard, uncomment and
# update the lines below.
# INCLFLAGS += -I/path/to/mlpack/include/
# INCLFLAGS += -I/path/to/ensmallen/include/
CXXFLAGS += $(INCLFLAGS)

OBJS1 := $(SRC1:.cpp=.o)
OBJS2 := $(SRC2:.cpp=.o)
LIBS := $(addprefix -l,$(LIBS_NAME))
CLEAN_LIST := $(TARGET1) $(TARGET2) $(OBJS1) $(OBJS2)

# default rule
default: all

$(TARGET1): $(OBJS1)
	$(CXX) $(CXXFLAGS) $(OBJS1) -o $(TARGET1) $(LDFLAGS) $(LIBS)

$(TARGET2): $(OBJS2)
	$(CXX) $(CXXFLAGS) $(OBJS2) -o $(TARGET2) $(LDFLAGS) $(LIBS)

.PHONY: all
all: $(TARGET1) $(TARGET2)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)
