use std::{cmp::Ordering};

mod matrix;
mod neural_net;

use rand::prelude::*;
extern crate roulette;

use roulette::Roulette;

use std::{thread, time};

fn main() {

    const POPULATION_SIZE: usize = 201;
    let mut population: Vec<neural_net::NeuralNet> = vec![];

    for _ in 0..POPULATION_SIZE {
        let nn = neural_net::NeuralNet::new (
            2,
            vec![
                neural_net::HiddenLayer::new (
                    5.0,
                    3,
                    None,
                    Some(-1.0..1.0),
                    neural_net::ActivationFunction::TanH // Having this as the identity function before prevented it from convering!
                ),
            ],
            neural_net::OutputLayer::new (
                3.0,
                1,
                None,
                Some(-1.0..1.0),
                neural_net::ActivationFunction::Sigmoid
            )
        );
        population.push(nn);
    };

    let mut generation = 0;
    loop {
        thread::sleep(time::Duration::from_millis(10));
        println!("generation {}", generation);

        let mut total_error = 0.0;
        let mut errors: Vec<(neural_net::NeuralNet, f64)> = vec![];
        for nn in population.iter() {
            let r1 = nn.forward(vec![0.0, 0.0]).get_row(0)[0]; // expect 0
            let r2 = nn.forward(vec![0.0, 1.0]).get_row(0)[0]; // expect 1
            let r3 = nn.forward(vec![1.0, 0.0]).get_row(0)[0]; // expect 1
            let r4 = nn.forward(vec![1.0, 1.0]).get_row(0)[0]; // expect 0

            let r1_diff = r1.abs();
            let r2_diff = (r2 - 1.0).abs();
            let r3_diff = (r3 - 1.0).abs();
            let r4_diff = r4.abs();

            if generation == 100 {
                println!("r1: {}", r1);
                println!("r2: {}", r2);
                println!("r3: {}", r3);
                println!("r4: {}", r4);
                panic!("Generation Limit Exceeded!");
            }

            let error = r1_diff + r2_diff + r3_diff + r4_diff;
            total_error += error;

            //println!("error: {error}");

            errors.push((nn.clone(), error));
        }

        population.clear();

        // Sort in ascending error according to the total error
        errors.sort_by(|(_, error_1), (_, error_2)| {
            if error_1 > error_2 { 
                Ordering::Greater 
            } else if error_1 < error_2 {
                Ordering::Less 
            } else {
                Ordering::Equal
            }
        });

        println!("lowest error: {}", errors[0].1);
        population.push(errors[0].0.clone());

        // Push on proportions to later pop off in reverse
        //  giving highest proportion to the smallest error (greatest fitness)
        let mut stack: Vec<f64> = vec![];
        for (_, error) in &errors {
            let p = error / total_error;
            stack.push(p);
        }

        let mut v: Vec<(neural_net::NeuralNet, f64)> = vec![];
        for (nn, _) in errors {
            let nn = nn.clone();
            let p = stack.pop().unwrap();

            v.push((nn, p));
        }

        // Roulette selection
        let mut rng = rand::thread_rng();
        let roulette = Roulette::new(v);
        for _ in 0..(POPULATION_SIZE - 1) {
            let rand1 = roulette.sample(&mut rng);
            let rand2 = roulette.sample(&mut rng);
            let mut new = neural_net::combine(rand1.clone(), rand2.clone());

            // apply mutation
            let r = rng.gen_range(0.0, 1.0);
            if r <= 0.05 { new.mutate() };
            population.push(new);
        }
        
        generation += 1;
    }

}
