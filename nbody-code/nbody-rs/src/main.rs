#![allow(unused_imports)]
mod nbody;
mod plotting;
use clap::Parser;
use nbody::{get_accel, get_energy, run_sim, NBodySystem, Tri};
use ndarray::{prelude::*, ShapeError};
use ndarray::{Array, Axis};

use color_eyre::eyre::Error;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
// use rand;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct CliArgParser {
    #[arg(short, long, default_value = "false")]
    images: bool,
    #[arg(short, long, default_value = "false")]
    video: bool,
}

// macro_rules! testmacro {
//     ($x:expr) => {{
//         let shape = ($x.len(), $x.len() + 1usize);
//         let fi = $x.broadcast(shape).unwrap().to_owned();
//         fi
//     }};
// }

// fn main() {
//     let a = Array1::<i32>::ones(3);
//     let b = testmacro!(a.clone().insert_axis(Axis(1)));
//     println!("a = \n{}\n", a);
//     println!("b = \n{}\n", b);
// }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgParser::parse();
    let n = 100; // Number of particles
    let t_end = 15.; // Time at which the sim ends
    let dt = 0.01; // Timestep
    let soft = 0.1; // softening length
    let g = 3.; // Newton's gravitational constant

    let mass = Array1::<f64>::from_elem(n, 20.); // total mass of the N particles
    let pos = Array2::<f64>::random((n, 3), Uniform::new(-4., 4.)); // random positions
    let vel = Array2::<f64>::random((n, 3), Uniform::new(-1., 1.));
    let accel = Array2::<f64>::zeros((n, 3));
    let mut nbsys = NBodySystem {
        n,
        g,
        soft,
        pos,
        mass,
        vel,
        accel,
    };
    let (pos, ke_save, pe_save, t_all) = run_sim(&mut nbsys, t_end, dt)?;

    if args.images {
        todo!()
    }

    std::mem::drop((pos, ke_save, pe_save, t_all)); // ! CHANGE BEFORE FINAL COMMIT

    Ok(())
}
