WDIR = .
SRC_DIR = $(WDIR)/src
OBJ_DIR = $(WDIR)/obj
ANALYSIS_DIR = $(WDIR)/analysis

CXX = /usr/local/cuda-12.8/bin/nvcc
GDB = /usr/local/cuda-12.8/bin/cuda-gdb
NCU = /usr/local/cuda-12.8/bin/ncu
NCU-GUI = /usr/local/cuda-12.8/bin/ncu-ui
NSYS = /usr/local/cuda-12.8/bin/nsys

TARGET = default
SRC = $(SRC_DIR)/$(TARGET).cu
EXEC = $(OBJ_DIR)/$(TARGET)
LDFLAGS += -lcublas
#CXXFLAGS += -g -G
#CXXFLAGS += -arch=sm_120

$(EXEC): $(SRC)
	@$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

run: $(EXEC)
	@./$<

gdb: $(EXEC)
	@$(GDB) $<


#Nsight System,系统级分析工具
nsys: $(EXEC)
	@$(NSYS) profile -t cuda,nvtx,osrt -o $(ANALYSIS_DIR)/$(TARGET) -f true $<
	@$(NSYS) stats $(ANALYSIS_DIR)/$(TARGET).nsys-rep --force-export=true

#Nsight Compute,内核级分析工具
ncu: $(EXEC)
	@$(NCU) --print-details all --nvtx --call-stack --set full $<

ncu-gui: $(EXEC)
	@$(NCU) --nvtx --call-stack --set full -f --export $(ANALYSIS_DIR)/$(TARGET).ncu-rep $<
	@$(NCU-GUI) $(ANALYSIS_DIR)/$(TARGET).ncu-rep


clean:
	@rm -r $(OBJ_DIR)/* $(ANALYSIS_DIR)/*

.PHONY:
