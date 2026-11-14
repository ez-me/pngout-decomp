# Reconstructed PNGOUT. The KZIP deflate engine lives in the same source
# directory, so a single target builds the whole tool.
CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic
LDFLAGS ?=

SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
HEADERS := $(wildcard src/*.hpp)

.PHONY: all clean
all: pngout

pngout: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJECTS)

src/%.o: src/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) pngout
