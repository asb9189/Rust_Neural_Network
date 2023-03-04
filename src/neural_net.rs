use core::num;
use std::ops::Range;

use crate::matrix::Matrix;

pub struct NeuralNet {
    num_input_nodes: usize,
    hidden_layers: Vec<HiddenLayer>,
    num_output_nodes: usize,
    output_layer: OutputLayer,
    weights: Vec<Matrix>
        // index 0 is the weight matrix for the first hidden layer
        // index 'len - 1' is the weight matrix for the output layer
}

impl NeuralNet {
    pub fn new(
        num_input_nodes: usize,
        num_output_nodes: usize,
        mut hidden_layers: Vec<HiddenLayer>,
        output_layer: OutputLayer
    ) -> NeuralNet {

        let mut weights: Vec<Matrix> = vec![];

        if num_input_nodes == 0 { panic!("NeuralNet cannot have zero inputs") };
        if num_output_nodes == 0 { panic!("NeuralNet cannot have zero output nodes") };

        let mut prev_node_count = num_input_nodes;
        for hl in hidden_layers.iter_mut() {
            let matrix = Matrix::from_range(
                hl.num_nodes,
                prev_node_count,
                (hl.weight_init_range.clone()).unwrap()
            );

            weights.push(matrix);
            prev_node_count = hl.num_nodes;
        }

        // Add output layer weights to NN weight matrix
        weights.push(Matrix::from_range(
            output_layer.num_nodes,
            num_input_nodes,
            (output_layer.weight_init_range.clone()).unwrap()
        ));

        NeuralNet {
            num_input_nodes: num_input_nodes,
            num_output_nodes: num_output_nodes,
            hidden_layers: hidden_layers,
            output_layer: output_layer,
            weights: weights,
        }

    }
}


pub struct HiddenLayer {
    bias: f64,
    num_nodes: usize,
    init_weights: Option<Vec<Matrix>>, // Not needed but nice to have
    weight_init_range: Option<Range<f64>>,
    activation_function: ActivationFunction
}

impl HiddenLayer {
    pub fn new (
        bias: f64,
        num_nodes: usize,
        init_weights: Option<Vec<Matrix>>,
        weight_init_range: Option<Range<f64>>,
        activation_function: ActivationFunction
    ) -> HiddenLayer {

        if init_weights.is_some() { panic!("Initial weights unimplemented!") }

        HiddenLayer { 
            bias: bias,
            num_nodes: num_nodes,
            init_weights: init_weights,
            weight_init_range: weight_init_range,
            activation_function: activation_function
        }

    }
}

pub struct OutputLayer {
    num_nodes: usize,
    weight_init_range: Option<Range<f64>>
}

impl OutputLayer {
    pub fn new(num_nodes: usize, range: Option<Range<f64>>) -> OutputLayer {
        if num_nodes == 0 { panic!("Output layer cannot have zero nodes") }

        OutputLayer { 
            num_nodes: num_nodes,
            weight_init_range: range
        }
    }
}

pub enum ActivationFunction {
    BinaryStep,
    Sigmoid,
    TanH
}
