WDIR = .
SRC_DIR = $(WDIR)/src
OBJ_DIR = $(WDIR)/obj
ANALYSIS_DIR = $(WDIR)/analysis

CXX = /usr/local/cuda-12.8/bin/nvcc
GDB = /usr/local/cuda-12.8/bin/cuda-gdb

TARGET = default
SRC = $(SRC_DIR)/$(TARGET).cu
EXEC = $(OBJ_DIR)/$(TARGET)
CXXFLAGS += -g -G

$(EXEC): $(SRC)
	@$(CXX) $(CXXFLAGS) -o $@ $<

run: $(EXEC)
	@./$<

gdb: $(EXEC)
	@$(GDB) $<


nsys: $(EXEC)
	@nsys profile -t cuda,nvtx,osrt -o $(ANALYSIS_DIR)/$(TARGET) -f true $<
	@nsys stats $(ANALYSIS_DIR)/$(TARGET).nsys-rep --force-export=true

clean:
	@rm -r $(OBJ_DIR)/* $(ANALYSIS_DIR)/*

.PHONY:
