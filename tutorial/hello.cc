#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"

#include <iostream>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);
  
  printf("Hello\n");


  return 0;
}
