#![allow(unused_imports, dead_code)]
mod nbody;
use clap::Parser;
use nbody::{get_accel, get_energy, run_sim, NBodySystem, Tri};
use ndarray::prelude::*;
use ndarray::{Array, Axis};
use rand;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct CliArgParser {
    #[arg(short, long)]
    save_images: bool,
}

fn main() {
    let a = Array::<f64, _>::ones((3, 3).f());
    let b = a.clone().tril(1);
    println!("{:?}", b);
    // println!("{:?}", a.len_of(Axis(1)))
}
// fn main() {
//     let n = 1000; // Number of particles
//     let t_end = 15.; // Time at which the sim ends
//     let dt = 0.01; // Timestep
//     let soft = 0.1; // softening length
//     let g = 3.; // Newton's gravitational constant

//     let mut rng = rand::thread_rng();
//     let mass = Array1::from_elem(n, 20. / n as f64); // total mass of the N particles
//     let pos = Array::from_shape_fn((n, 3), |_| rng.gen_range(-4.0..4.0)); // random positions
//     let vel = Array::from_shape_fn((n, 3), |_| rng.gen_range(-1.0..1.0));
//     let accel = Array::zeros((n, 3));

//     let mut nbsys = NBodySystem {
//         n,
//         g,
//         soft,
//         pos,
//         mass,
//         vel,
//         accel,
//     };

//     let (pos, ke_save, pe_save, t_all) = run_sim(&mut nbsys, t_end, dt);
// }
