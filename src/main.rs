mod matrix;
mod neural_net;

fn main() {
    let nn = neural_net::NeuralNet::new (
        2,
        vec![
            neural_net::HiddenLayer::new (
                0.0,
                3,
                Some(matrix::Matrix::from(vec![
                    vec![1.0, 2.0, 3.0],
                    vec![4.0, 5.0, 6.0]
                ])),
                None,
                neural_net::ActivationFunction::Identity
            ),
        ],
        neural_net::OutputLayer::new (
            0.0,
            1,
            Some(matrix::Matrix::from(vec![
                vec![1.0],
                vec![2.0],
                vec![3.0]
            ])),
            None,
            neural_net::ActivationFunction::Identity
        )
    );

    nn.display();

    let r = nn.forward(vec![-5.0, 5.0]);
    println!("##### Result #####");
    r.display();
}
