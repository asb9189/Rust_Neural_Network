use core::num;
use std::{ops::Range, f64::consts::E};

use crate::matrix::{Matrix, self};

pub struct NeuralNet {
    num_input_nodes: usize,
    num_output_nodes: usize,
    hidden_layers: Vec<HiddenLayer>,
    output_layer: OutputLayer,
    activation_functions: Vec<ActivationFunction>,
    weights: Vec<Matrix>
        // index 0 is the weight matrix for the first hidden layer
        // index 'len - 1' is the weight matrix for the output layer
}

impl NeuralNet {
    pub fn new (
        num_input_nodes: usize,
        mut hidden_layers: Vec<HiddenLayer>,
        output_layer: OutputLayer
    ) -> NeuralNet {

        let mut weights: Vec<Matrix> = vec![];

        if num_input_nodes == 0 { panic!("NeuralNet cannot have zero inputs") };
        if output_layer.num_nodes == 0 { panic!("NeuralNet cannot have zero output nodes") };

        let mut afs: Vec<ActivationFunction> = vec![];
        let mut prev_node_count = num_input_nodes;
        for hl in hidden_layers.iter_mut() {
            let matrix = Matrix::from_range(
                prev_node_count,
                hl.num_nodes,
                (hl.weight_init_range.clone()).unwrap()
            );

            weights.push(matrix);
            afs.push(hl.activation_function);
            prev_node_count = hl.num_nodes;
        }

        // Add output layer weights to NN weight matrix
        afs.push(output_layer.activation_function);
        weights.push(Matrix::from_range(
            prev_node_count,
            output_layer.num_nodes,
            (output_layer.weight_init_range.clone()).unwrap()
        ));

        let l = weights.len();
        println!("Matrix has {l} weight matrices");

        assert!(weights.len() == afs.len(), "internal error creating neural network");

        NeuralNet {
            num_input_nodes: num_input_nodes,
            num_output_nodes: output_layer.num_nodes,
            hidden_layers: hidden_layers,
            output_layer: output_layer,
            activation_functions: afs,
            weights: weights,
        }

    }

    pub fn forward(&self, input: Vec<f64>) -> Matrix {

        if input.len() != self.num_input_nodes { panic!("Invalid number of inputs") };

        let mut result = Matrix::new(1, 1);
        let mut input = Matrix::from(vec![input]);
        for i in 0..(self.weights.len()) {
            result = matrix::mul(&input, &self.weights[i]);
            result.map(|x| {
                activate(x, self.activation_functions[i])
            });

            // After 'mul' add bias to each value from layer's bias
            //  and wrap in layer's associated activation function
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
    activation_function: ActivationFunction,
    weight_init_range: Option<Range<f64>>
}

impl OutputLayer {
    pub fn new(num_nodes: usize, range: Option<Range<f64>>, af: ActivationFunction) -> OutputLayer {
        if num_nodes == 0 { panic!("Output layer cannot have zero nodes") }

        OutputLayer { 
            num_nodes: num_nodes,
            activation_function: af,
            weight_init_range: range,
        }
    }
}

#[derive(Copy, Clone)]
pub enum ActivationFunction {
    Identity,
    Step(f64),
    BinaryStep,
    Sigmoid,
    TanH,
}

fn activate(x: f64, af: ActivationFunction) -> f64 {
    match af {
        ActivationFunction::Identity => x,
        ActivationFunction::Step(t) => step(x, t),
        ActivationFunction::BinaryStep => binary_step(x),
        ActivationFunction::Sigmoid => sigmoid(x),
        ActivationFunction::TanH => tanh(x),
    }
}

fn step(x: f64, t: f64) -> f64 {
    if x > t { 1.0 } else { 0.0 }
}

fn binary_step(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

// https://en.wikipedia.org/wiki/Sigmoid_function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

// https://www.mathworks.com/help/matlab/ref/tanh.html
fn tanh(x: f64) -> f64 {
    (E.powf(2.0 * x) - 1.0) / (E.powf(2.0 * x) + 1.0)
}
