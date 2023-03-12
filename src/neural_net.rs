use std::{ops::Range, f64::consts::E};
use crate::matrix::{Matrix, self};

pub struct NeuralNet {

    // Number of input nodes for the neural network
    num_input_nodes: usize,

    // where index 0 is the activation function for the first
    //  hidden layer and index (len - 1) is the af for the output layer
    activation_functions: Vec<ActivationFunction>,

    // where index 0 is the weight matrix for the first hidden layer
    //  and index (len - 1) is the weight matrix for the output layer
    weights: Vec<Matrix>
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
            if hl.init_weights.is_some() {
                let matrix = hl.init_weights.as_ref().unwrap().clone();
                
                // check that given initial weights are valid
                if matrix.num_rows() != prev_node_count { panic!("Invalid initial weights matrix given") }
                if matrix.num_cols() != hl.num_nodes { panic!("Invalid initial weights matrix given") }

                weights.push(matrix);
            } else {
                let matrix = Matrix::from_range(
                    prev_node_count,
                    hl.num_nodes,
                    (hl.weight_init_range.clone()).unwrap()
                );
                weights.push(matrix);
            }
            afs.push(hl.activation_function);
            prev_node_count = hl.num_nodes;
        }

        // Add output layer weights to NN weight matrix
        afs.push(output_layer.activation_function);

        if output_layer.init_weights.is_some() {
            let matrix = output_layer.init_weights.as_ref().unwrap().clone();

            // check that given initial weights are valid
            if matrix.num_rows() != prev_node_count { panic!("Invalid initial output weights matrix given") }
            if matrix.num_cols() != output_layer.num_nodes { panic!("Invalid initial output weights matrix given") }

            weights.push(matrix);
        } else {
            weights.push(Matrix::from_range(
                prev_node_count,
                output_layer.num_nodes,
                (output_layer.weight_init_range.clone()).unwrap()
            ));
        }

        assert!(weights.len() == afs.len(), "internal error creating neural network");

        NeuralNet {
            num_input_nodes: num_input_nodes,
            activation_functions: afs,
            weights: weights,
        }

    }

    pub fn display(&self) {

        println!("##### Weight Matrices of Neural Network #####");

        let mut i = 0;
        let last_i = &self.weights.len() - 1;
        for wm in &self.weights {
            if i == last_i {
                println!("output layer:\n");
                wm.display();
            } else {
                println!("{i}th hidden layer:\n");
                wm.display();
            }
            i += 1;
        }
        
        println!("#############################################");
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
    init_weights: Option<Matrix>,
    weight_init_range: Option<Range<f64>>,
    activation_function: ActivationFunction
}

impl HiddenLayer {
    pub fn new (
        bias: f64,
        num_nodes: usize,
        init_weights: Option<Matrix>,
        weight_init_range: Option<Range<f64>>,
        activation_function: ActivationFunction
    ) -> HiddenLayer {

        if num_nodes == 0 { panic!("Cannot create hidden layer with zero nodes") }

        match init_weights {
            Some(iw) => {
                match weight_init_range {
                    Some(_) => panic!("cannot apply weight_init_range to given init_weights"),
                    None => {
                        HiddenLayer {
                            bias: bias,
                            num_nodes: num_nodes,
                            init_weights: Some(iw),
                            weight_init_range: None,
                            activation_function: activation_function
                        }
                    }
                }
            },
            None => {
                match weight_init_range {
                    Some(range) => {
                        HiddenLayer { 
                            bias: bias,
                            num_nodes: num_nodes,
                            init_weights: None,
                            weight_init_range: Some(range),
                            activation_function: activation_function
                        }
                    },
                    None => panic!("weight_init_range necessary if init_weights not given")
                }
            }
        }
    }
}

pub struct OutputLayer {
    num_nodes: usize,
    activation_function: ActivationFunction,
    weight_init_range: Option<Range<f64>>,
    init_weights: Option<Matrix>,
}

impl OutputLayer {
    pub fn new(num_nodes: usize, init_weights: Option<Matrix>, range: Option<Range<f64>>, af: ActivationFunction) -> OutputLayer {
        if num_nodes == 0 { panic!("Output layer cannot have zero nodes") }

        match init_weights {
            Some(iw) => {
                match range {
                    Some(_) => panic!("cannot apply weight_init_range to given init_weights"),
                    None => {
                        OutputLayer { 
                            num_nodes: num_nodes,
                            activation_function: af,
                            weight_init_range: range,
                            init_weights: Some(iw)
                        }
                    }
                }
            },
            None => {
                match range {
                    Some(range) => {
                        OutputLayer { 
                            num_nodes: num_nodes,
                            activation_function: af,
                            weight_init_range: Some(range),
                            init_weights: None
                        }
                    },
                    None => panic!("weight_init_range necessary if init_weights not given")
                }
            }
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

// ###################################################################################################

#[cfg(test)]
mod tests {
    
    use crate::matrix::Matrix;
    use super::{NeuralNet, HiddenLayer, OutputLayer, ActivationFunction};

    #[test]
    fn build_nn_00() {
        NeuralNet::new (
            1,
            vec![],
            OutputLayer::new(
                1,
                None,
                Some(-1.0..1.0),
                ActivationFunction::Identity
            ),
        );
    }

    #[test]
    fn build_nn_01() {
        NeuralNet::new (
            1,
            vec![
                HiddenLayer::new(
                    0.0,
                    5,
                    None,
                    Some(-1.0..1.0),
                    ActivationFunction::Identity
                )
            ],
            OutputLayer::new(
                1,
                None,
                Some(-1.0..1.0),
                ActivationFunction::Identity
            ),
        );
    }

    #[test]
    fn build_nn_02() {
        NeuralNet::new (
            1,
            vec![
                HiddenLayer::new(
                    0.0,
                    5,
                    None,
                    Some(-1.0..1.0),
                    ActivationFunction::Identity
                ),
                HiddenLayer::new(
                    0.0,
                    3,
                    None,
                    Some(-1.0..1.0),
                    ActivationFunction::Identity
                ),
            ],
            OutputLayer::new(
                1,
                None,
                Some(-1.0..1.0),
                ActivationFunction::Identity
            ),
        );
    }

    #[test]
    fn forward_nn_00() {
        let nn = NeuralNet::new (
            2,
            vec![
                HiddenLayer::new(
                    0.0,
                    3,
                    Some(Matrix::from(vec![
                        vec![1.0, 2.0, 3.0],
                        vec![4.0, 5.0, 6.0]
                    ])),
                    None,
                    ActivationFunction::Identity
                ),
            ],
            OutputLayer::new (
                1,
                Some(Matrix::from(vec![
                    vec![1.0],
                    vec![2.0],
                    vec![3.0]
                ])),
                None,
                ActivationFunction::Identity
            )
        );

        let result = nn.forward(vec![-5.0, 5.0]);
        assert!(result.num_rows() == 1);
        assert!(result.num_cols() == 1);

        assert!((result.get_row(0))[0] == 90.0);
    }

    #[test]
    #[should_panic]
    fn build_fail_nn_00() {
        NeuralNet::new (
            0,
            vec![],
            OutputLayer::new(
                1,
                None,
                Some(-1.0..1.0),
                ActivationFunction::Identity
            ),
        );
    }

    #[test]
    #[should_panic]
    fn build_fail_nn_01() {
        NeuralNet::new (
            1,
            vec![],
            OutputLayer::new(
                0,
                None,
                Some(-1.0..1.0),
                ActivationFunction::Identity
            ),
        );
    }
}
