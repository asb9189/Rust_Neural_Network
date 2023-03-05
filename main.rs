use neural_net::OutputLayer;

mod matrix;
mod neural_net;

fn main() {
    let nn = neural_net::NeuralNet::new(
        2,
        5,
        vec![
            neural_net::HiddenLayer::new(0.0,
                4,
                None,
                Some(0.0..10.0),
                neural_net::ActivationFunction::Sigmoid
            )
        ],
        neural_net::OutputLayer::new (
            5,
            Some(0.0..10.0)
        )
    );

    let r = nn.forward(vec![1.0, 2.0]);

    println!("##### Result #####");
    r.display();
}
