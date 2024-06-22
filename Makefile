# Compiler and flags
CC = gcc
CFLAGS = -Wall -Werror -Iinclude

# Directories
SRCDIR = src
INCDIR = include
TESTDIR = test
BINDIR = bin

# Source files and object files
SOURCES = $(SRCDIR)/matrix.c
OBJECTS = $(SOURCES:.c=.o)
TESTS = $(TESTDIR)/test_matrix.c

# Executable names
LIB = libmatrix.a
TEST_EXE = test_matrix

# Targets
.PHONY: all clean test

all: $(BINDIR)/$(LIB)

# Create the static library
$(BINDIR)/$(LIB): $(OBJECTS)
	@mkdir -p $(BINDIR)
	ar rcs $@ $^

# Compile object files
$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Run tests
test: $(TESTDIR)/$(TEST_EXE)
	./$(TESTDIR)/$(TEST_EXE)

# Build test executable
$(TESTDIR)/$(TEST_EXE): $(TESTS) $(BINDIR)/$(LIB)
	$(CC) $(CFLAGS) -o $@ $(TESTS) -L$(BINDIR) -lmatrix -lm

clean:
	rm -f $(SRCDIR)/*.o $(BINDIR)/* $(TESTDIR)/$(TEST_EXE)

