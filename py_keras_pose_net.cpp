/*
 *  Copyright 2018 Digital Media Professionals Inc.

 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at

 *      http://www.apache.org/licenses/LICENSE-2.0

 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.

 *  This source code was generated using DMP-DV700 tools.
 */

/**
* @file py_keras_pose_net.cpp
* @brief collection of abstract interfaces to CKerasPoseNet
*/

#include <vector>
#include <stdexcept>
#include <boost/python/numpy.hpp>

#include "KerasPoseNet_gen.h"


namespace py = boost::python;
namespace np = py::numpy;

using net_id = unsigned int;
using namespace std;
using CNet = CKerasPoseNet;

namespace {
	vector<CNet*> nets(1);
}

net_id create(void)
{
	CNet * net = new CNet();
	nets.push_back(net);

	return nets.size() - 1;
}

bool initialize(net_id id)
{
	return nets[id]->Initialize();
}

bool load_weights(net_id id, const string& filename)
{
	return nets[id]->LoadWeights(filename);
}

bool commit(net_id id)
{
	return nets[id]->Commit();
}

bool run_network(net_id id)
{
	return nets[id]->RunNetwork();
}

/**
 * @brief pass input to the CNet
 * @param[in] (id) ID of CNet network
 * @param[in] (input) input to the network. Type of input must be np.float16.
 		Call np.ascontiguousarray(input) in advance.
 */
void put_input(net_id id, const np::ndarray & input)
{
	CNet * net = nets[id];

	// - convert the ndarray into an array
	if (input.get_dtype() != np::dtype::get_builtin<__fp16>())
	{
		throw runtime_error("input array must be fp16 array");
	}
	__fp16 * data = reinterpret_cast<__fp16 *>(input.get_data());

	// get input size
	int inlen = 1;
	int dim = input.get_nd();
	for(int i = 0; i < dim; i++)
	{
		inlen *= input.shape(i);
	}

	// copy the input data
	void * addr_input_buf = net->get_network_input_addr_cpu();
	memcpy(addr_input_buf, data, inlen * sizeof(__fp16));
}

/**
 * @brief get output of the CNet
 * @param[in] id ID of CNet network
 * @param[in] index index of output
 *
 * @return boost::python::numpy::ndarray An output from CNet network.
 */
np::ndarray get_final_output(net_id id, int index = 0)
{
	CNet * net = nets[id];

	// get output
	vector<float> * output = new std::vector<float>();
	net->get_final_output(*output, index);

	// convert the vector to ndarray
	static const np::dtype ftdt = np::dtype::get_builtin<float>();
	const vector<int> shape(1, output->size());
	const vector<int> stride(1, sizeof(float));
	py::object own;
	return np::from_data(&(*output)[0], ftdt, shape, stride, own);
}

int get_conv_usec(net_id id)
{
	return nets[id]->get_conv_usec();
}

int get_fc_usec(net_id id)
{
	return nets[id]->get_fc_usec();
}

int get_cpu_usec(net_id id)
{
	return nets[id]->get_cpu_usec();
}

void destroy(net_id id)
{
	delete nets[id];
}

BOOST_PYTHON_FUNCTION_OVERLOADS(f_overloads, get_final_output, 1, 2)

BOOST_PYTHON_MODULE(py_keras_pose_net) {
	Py_Initialize();
	np::initialize();

	// export the functions
	py::def("create", create);
	py::def("initialize", initialize);
	py::def("load_weights", load_weights);
	py::def("commit", commit);
	py::def("run_network", run_network);
	py::def("put_input", put_input);
	py::def("get_final_output", get_final_output, f_overloads(
		py::args("id", "index")));
	py::def("get_conv_usec", get_conv_usec);
	py::def("get_fc_usec", get_fc_usec);
	py::def("get_cpu_usec", get_cpu_usec);
	py::def("destroy", destroy);
}
