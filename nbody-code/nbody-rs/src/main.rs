mod nbody;
mod plotting;
use std::fs::create_dir;
use std::path::Path;

use clap::Parser;
use color_eyre::eyre::{Error, Result};

use nbody::{run_sim, NBodySystem};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use plotting::plot_nbodysystem;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct CliArgParser {
    #[arg(short, long, default_value = "false")]
    video: bool,
}

fn main() -> Result<(), Error> {
    let args = CliArgParser::parse();
    let n = 100; // Number of particles
    let t_end = 10.; // Time at which the sim ends
    let dt = 0.1; // 0.01 Timestep
    let soft = 0.1; // softening length
    let g = 3.; // Newton's gravitational constant

    let mass = Array1::<f64>::from_elem(n, 20.); // total mass of the N particles
    let pos = Array2::<f64>::random((n, 3), Normal::new(0., 2.)?); // random positions
    let vel = Array2::<f64>::random((n, 3), Normal::new(0., 1.)?);
    // let pos = Array2::<f64>::random((n, 3), Uniform::new(-4., 4.)); // random positions
    // let vel = Array2::<f64>::random((n, 3), Uniform::new(-1., 1.));
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

    let (pos, ke, pe, t_all) = run_sim(&mut nbsys, t_end, dt)?;

    let dir = "images";

    let dirpath = Path::new(&dir);
    if !dirpath.exists() {
        create_dir(dirpath)?
    }

    let filename = format!("./{dir}/nbsys").to_string();

    plot_nbodysystem(pos, ke, pe, t_all, filename)?;

    if args.video {
        todo!()
    }
    Ok(())
}
