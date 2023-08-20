mod nbody;
mod plotting;
use clap::Parser;
use color_eyre::eyre::Result;
use nbody::{run_sim, NBodySystem};
use ndarray::{prelude::*, ShapeError};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct CliArgParser {
    #[arg(short, long, default_value = "false")]
    images: bool,
    #[arg(short, long, default_value = "false")]
    video: bool,
}

fn main() -> Result<(), ShapeError> {
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
