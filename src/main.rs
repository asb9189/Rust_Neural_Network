use neural_net::OutputLayer;

mod matrix;
mod neural_net;

fn main() {
    let r = matrix::Matrix::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0]
    ]);

    r.display();
    
    let nn = neural_net::NeuralNet::new(
        1,
        1,
        vec![
            neural_net::HiddenLayer::new(0.0,
                4,
                None,
                Some(0.0..10.0),
                neural_net::ActivationFunction::Sigmoid
            )
        ],
        neural_net::OutputLayer::new (
            1,
            Some(0.0..10.0)
        )
    );

    
}
