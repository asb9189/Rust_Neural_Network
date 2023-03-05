use core::num;
use std::ops::Range;

use crate::matrix::{Matrix, self};

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
    pub fn new (
        num_input_nodes: usize,
        num_output_nodes: usize,
        mut hidden_layers: Vec<HiddenLayer>,
        output_layer: OutputLayer
    ) -> NeuralNet {

        let mut weights: Vec<Matrix> = vec![];

        if num_input_nodes == 0 { panic!("NeuralNet cannot have zero inputs") };
        if num_output_nodes == 0 { panic!("NeuralNet cannot have zero output nodes") };

        // TODO fix dimensions of each matrix in weight matrix
        let mut prev_node_count = num_input_nodes;
        for hl in hidden_layers.iter_mut() {
            let matrix = Matrix::from_range(
                prev_node_count,
                hl.num_nodes,
                (hl.weight_init_range.clone()).unwrap()
            );

            weights.push(matrix);
            prev_node_count = hl.num_nodes;
        }

        // Add output layer weights to NN weight matrix
        weights.push(Matrix::from_range(
            prev_node_count,
            output_layer.num_nodes,
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

    pub fn forward(&self, input: Vec<f64>) -> Matrix {

        if input.len() != self.num_input_nodes { panic!("Invalid number of inputs") };

        let mut result = Matrix::new(1, 1);
        let mut input = Matrix::from(vec![input]);
        for i in 0..(self.weights.len()) {
            result = matrix::mul(&input, &self.weights[i]);
            input = result.clone();
        }

        result
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
