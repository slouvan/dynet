#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/dict.h"
#include "dynet/expr.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace dynet;

using namespace std;
using namespace dynet;
using namespace dynet::expr;

/*
debug
void Print (const vector<float>& v){
  //vector<int> v;
  for (int i=0; i<v.size();i++){
    cout << v[i] << endl;
  }
}*/



/*vector<string> read_lines(string file_name)
{
	vector<string> lines;
	ifstream myfile (file_name);
	string line;
	while (getline(myfile, line))
	{     
		lines.push_back(line);
	}
}*/


int main(int argc, char** argv) {
	dynet::initialize(argc, argv); // need to add this apparently ??
	
	dynet::Dict d;
	//int start_sent 	= d.convert("<s>");
  	//int end_sent 	= d.convert("</s>");

  	// Read data
	vector<vector<int>> training;
	
	cout << "READING DATA" << endl;
	ifstream in("/home/slouvan/dynet/examples/example-data/fin-words.txt");
	string line;
	while(getline(in, line)) {
      training.push_back(read_sentence(line, &d));
      cout<< line << endl;
    }


	unsigned hidden_dim 		= 50;
	unsigned layer_input_dim 	= 10;
	unsigned VOCAB_SIZE 		= d.size();

	// Define parameters
	Model model;
	LookupParameter p_c = model.add_lookup_parameters(VOCAB_SIZE, {layer_input_dim}); 
	Parameter 		p_W = model.add_parameters({hidden_dim, layer_input_dim});
	Parameter 		p_U	= model.add_parameters({hidden_dim, hidden_dim});
	Parameter 		p_B = model.add_parameters({hidden_dim});
	Parameter 		p_V = model.add_parameters({VOCAB_SIZE, hidden_dim});
	
	// Add parameters to the graph
	ComputationGraph cg;
  	Expression W = parameter(cg, p_W);
  	Expression U = parameter(cg, p_U);
  	Expression b = parameter(cg, p_B);
  	Expression V = parameter(cg, p_V);
  	
  	vector<Expression> h_t; // store previous hidden state
  	
  	SimpleSGDTrainer trainer(&model, 0.1);
  	

  	// TRAINING
  	
  	for (unsigned i = 0; i < training.size(); ++i) {
  		vector<int> current_sent = training[i];

  		// for each training instance build the sequence
  		vector<Expression> h_t_1;
  		Expression h0; // how to init the first one?
  		for (unsigned t = 0; t < current_sent.size(); ++t)
  		{ 
  			Expression x_t = lookup(cg, p_c, current_sent[t]);	// input
  			Expression h_t;

  			if (t == 0) // the first time step
  			{
  				h_t = tanh(W * x_t + U * h0 + b);
  			}
  			else
  			{
  				h_t = tanh(W * x_t + U * h_t_1[t-1] + b);
  			}

  			// compute and sum error here
  			
  			h_t_1.push_back(h_t); // store the hidden state
  		}

  		
  		// Expression loss = 
  		// cg.forward(loss);
  		// cg.backward(loss);
  		// trainer.update();
  		
  	}
	return 0;
}
