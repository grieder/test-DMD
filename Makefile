DIRBIN = ./build
TARGET = $(DIRBIN)/$(EXE)
EXE = DMD
SRC_DIRS = ./src

all: $(TARGET)

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)
OBJS := $(addsuffix .o,$(basename $(SRCS)))
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS = $(INC_FLAGS) -MMD -MP

$(TARGET): $(OBJS)
	$(CC) $(CPPFLAGS) $(IFLAGS) $(LDFLAGS) $(OBJS) -o $@ $(LOADLIBES) $(LDLIBS)

$(OBJS) : $(SRC_DIRS)/%.o: $(SRC_DIRS)/%.cpp
	$(CC) $(CPPFLAGS) $(IFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -rf $(TARGET) $(OBJS) $(DEPS)

-include $(DEPS) ./make.inc

rm_d_files:
	$(shell rm $(SRC_DIRS)/*.d)
