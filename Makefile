CC = gcc

CFLAGS = -Wall -Wextra -O2 -g -I. -std=c99
LDFLAGS =
LDLIBS = -lm -lgsl -lgslcblas

SRCS_LIB = cmaes.c optimizer.c
OBJS_LIB = $(SRCS_LIB:.c=.o)

SRCS_BENCH = main_benchmark.c
OBJS_BENCH = $(SRCS_BENCH:.c=.o)

TARGET_BENCH = benchmark_test


.PHONY: all
all: $(TARGET_BENCH)

$(TARGET_BENCH): $(OBJS_BENCH) $(OBJS_LIB)
	@echo "Linking $@..."
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)
	@echo "Benchmark executable '$(TARGET_BENCH)' created successfully."
	@echo "Run with: ./$(TARGET_BENCH) [function_index] [dimension]"
	@echo "Example:  ./$(TARGET_BENCH) 2 10  (Run Rastrigin in 10D)"
	@echo "Run './$(TARGET_BENCH) -h' for function list."


%.o: %.c $(wildcard *.h)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@


.PHONY: lib_objects
lib_objects: $(OBJS_LIB)
	@echo "Library object files created: $(OBJS_LIB)"


.PHONY: clean
clean:
	@echo "Cleaning up object files and executables..."
	rm -f $(OBJS_LIB) $(OBJS_BENCH) $(TARGET_BENCH)


# LIB_TARGET = libcmaesopt.a
#
# .PHONY: library
# library: $(LIB_TARGET)
#
# $(LIB_TARGET): $(OBJS_LIB)
#	@echo "Creating static library $@..."
#	ar rcs $@ $^
#	@echo "Static library '$@' created successfully."
#
# clean:
#	rm -f $(LIB_TARGET)
