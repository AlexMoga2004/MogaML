# Makefile for my_project

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Werror -Iinclude

# Linker flags
LDFLAGS = 

# Source and object files
SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)

# Static library for Matrix
LIB_MATRIX = lib/libmatrix.a

# Target executable
TARGET = my_project

# Build target
all: $(TARGET)

# Link objects to create executable
$(TARGET): $(OBJ) $(LIB_MATRIX)
	$(CC) -o $@ $^ $(LDFLAGS)

# Compile source files to object files
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Build static library for Matrix
$(LIB_MATRIX): src/matrix.o
	ar rcs $@ $^

# Clean project
clean:
	rm -f $(OBJ) $(TARGET) $(LIB_MATRIX)

# Test target
test: $(TARGET)
	@echo "Running tests..."
	@./$(TARGET)

# Phony targets
.PHONY: all clean test

